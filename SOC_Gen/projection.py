import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from ccdiff.layers import PositionNet, GatedSelfAttentionDense

class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs=64, temperature=100):
        super().__init__()

        self.num_freqs = num_freqs
        self.temperature = temperature

        freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)
        freq_bands = freq_bands[None, None]
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def __call__(self, x):
        x = self.freq_bands * x.unsqueeze(-1)
        return torch.stack((x.sin(), x.cos()), dim=-1).permute(0, 2, 3, 1).reshape(x.shape[0], -1)

def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x

class SelfAttentionLayer(nn.Module):
    def __init__(self, channels, nhead, dropout=0.0):
        super().__init__() 
        self.norm1 = nn.LayerNorm(channels)
        self.self_attn = nn.MultiheadAttention(channels, nhead, dropout=dropout)

        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                input,
                mask = None,):
        h = self.norm1(input)
        h1 = self.self_attn(query=h, key=h, value=h, attn_mask=mask)[0]
        h = h + self.dropout(h1)
        h = self.norm2(h)
        return h

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)

class CrossAttentionLayer(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.perceiver_fg = PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads)
        self.perceiver_bg = PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads)
    def forward(self, x, latents):
        x_fg, x_bg = x
        latents_fg, latents_bg = latents
        out_fg = self.perceiver_fg(x_fg, latents_fg)
        out_bg = self.perceiver_bg(x_bg, latents_bg)
        return out_fg, out_bg

class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        
        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim)
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
    
    def forward(self, x, context):
        x = self.norm(x)
        context = self.context_norm(context)
        
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        # print("q shape:", q.shape)
        # print("k shape:", k.shape)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))
        
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
        return self.to_out(out)

class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        max_seq_len: int = 257,  # CLIP tokens + CLS token
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,  # number of latents derived from mean pooled representation of the sequence
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None # 若apply_pos_emb为True，则初始化位置嵌入层pos_emb

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5) # 初始化可学习的潜在向量

        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x, coherent_queries=None):
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            x = x + pos_emb

        latents = self.latents.repeat(x.size(0), 1, 1) if coherent_queries is None else \
                  torch.cat([self.latents, coherent_queries], dim=1).repeat(x.size(0), 1, 1)

        x = self.proj_in(x)
        
        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        
        latents = self.proj_out(latents)
        return self.norm_out(latents)

class EnhancedResampler(Resampler):
    def __init__(self, controlnet_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.controlnet_dim = controlnet_dim
        
        if controlnet_dim:
            self.ctrl_attn = CrossAttention(
                dim=kwargs['dim'],
                context_dim=controlnet_dim,
                dim_head=kwargs['dim_head'],
                heads=kwargs['heads']
            )
            self.ctrl_gate = nn.Parameter(torch.zeros(1))
        else:
            self.ctrl_attn = None

    def forward(self, x, coherent_queries=None, controlnet_feats=None):
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            x = x + pos_emb

        latents = self.latents.expand(x.size(0), -1, -1).contiguous() if coherent_queries is None else \
            torch.cat([self.latents.expand(x.size(0), -1, -1), coherent_queries], dim=1)
        
        x = self.proj_in(x)
        
        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)
        
        if controlnet_feats is not None and self.ctrl_attn is not None:
            if controlnet_feats.shape[0] != latents.shape[0]:
                controlnet_feats = controlnet_feats.expand(latents.shape[0], -1, -1).contiguous()
            ctrl_out = self.ctrl_attn(
                latents, 
                context=controlnet_feats
            )
            latents = latents + self.ctrl_gate * ctrl_out
        
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        
        latents = self.proj_out(latents)
        return self.norm_out(latents)

class Controller(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        num_queries=[8, 8, 8],
        embedding_dim=768,
        output_dim=1024,
        controlnet_dim=1024,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.fg_resampler = EnhancedResampler(
            dim=dim,
            depth=depth,
            heads=dim // dim_head,
            dim_head=dim_head,
            num_queries=num_queries[0],
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            controlnet_dim=controlnet_dim,
            **kwargs
        )
        self.bg_resampler = EnhancedResampler(
            dim=dim,
            depth=depth,
            heads=dim // dim_head,
            dim_head=dim_head,
            num_queries=num_queries[1],
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            controlnet_dim=controlnet_dim,
            **kwargs
        )
        self.point_net = PositionNet(
            in_dim=output_dim,
            out_dim=output_dim
        )
        self.coherent_bridge = GatedSelfAttentionDense(
            query_dim=dim,
            context_dim=output_dim,
            n_heads=dim // dim_head,
            d_head=dim_head
        )
        self.coherent_queries = nn.Parameter(torch.randn(1, num_queries[2], dim) / dim**0.5)
        
    def forward(self, x_objs, obboxes, x_bg, x_condition=None):
        B = x_bg.shape[0]
        obboxes = torch.from_numpy(np.array([obbox[::2] + obbox[1::2] for obbox in obboxes[0]])).float().to(x_objs.device)
        embed_obboxes = self.point_net(obboxes).unsqueeze(1)
    
        embed_objs = self.fg_resampler(
            x_objs,
            controlnet_feats=x_condition
        )

        coherent_queries = self.coherent_bridge(
            self.coherent_queries,
            (embed_obboxes + embed_objs.detach()).view(B, -1, self.output_dim)
        )
        
        embed_context = self.bg_resampler(
            x_bg,
            coherent_queries,
            controlnet_feats=x_condition
        )
        return embed_objs, embed_context
        

def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)
