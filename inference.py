from typing import Any, Callable, Dict, List, Optional, Union
from diffusers import EulerDiscreteScheduler, AutoencoderKL, ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.loaders import AttnProcsLayers
import os, json
import torch
import time
from torchvision import transforms
from safetensors.torch import load_file
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.dior_utils import logging
from PIL import Image
from transformers import (
    CLIPImageProcessor, 
    CLIPTextModel, 
    CLIPTokenizer, 
    CLIPTextModelWithProjection, 
    CLIPVisionModelWithProjection, 
)
import inspect
from SOC_Gen.attention_processor import set_processors
from SOC_Gen.dior_utils import dict_of_images, find_nearest, get_similar_examplers
from SOC_Gen.projection import Controller
from SOC_Gen.modules import FrozenDinoV2Encoder

import diffusers.models.attention_processor
diffusers.models.attention_processor.logger.setLevel("ERROR")

logger = logging.get_logger(__name__)

class CustomControlNetModel(ControlNetModel):
    @classmethod
    def from_unet(
        cls,
        unet,
        load_weights=True,
        **kwargs,
    ):
        unet_config = dict(unet.config)

        init_signature = inspect.signature(cls.__init__)
        allowed_params = set(init_signature.parameters.keys())
        filtered_config = {
            k: v for k, v in unet_config.items() if k in allowed_params
        }
        controlnet = cls(**filtered_config)

        if load_weights:
            controlnet.mid_block.load_state_dict(
                unet.mid_block.state_dict(), 
                strict=False
            )
            # process down_blocks
            for i, down_block in enumerate(controlnet.down_blocks):
                down_block.load_state_dict(
                    unet.down_blocks[i].state_dict(),
                    strict=False
                )
        return controlnet

class StableDiffusionMIPipeline(StableDiffusionControlNetPipeline):
    _optional_components = [
        "safety_checker",
        "image_encoder"
    ]
    _config = [
        "vae",
        "text_encoder",
        "tokenizer",
        "unet",
        "controlnet",
        "scheduler",
        "feature_extractor",
        "safety_checker",
        "image_encoder",
    ]
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        image_encoder = FrozenDinoV2Encoder()
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )
        self.text_projector = CLIPTextModelWithProjection.from_pretrained(
            'path/to/clip'
        ).to(self.device)

        self.image_encoder.to(self.device)

        self.controlnet.to(self.device)

        self.My_Proj_Model = Controller(
            dim=1280,
            depth=4,
            dim_head=64,
            num_queries=[16, 8, 8],
            embedding_dim=self.image_encoder.model.embed_dim,
            output_dim=unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device)
    
    def _encode_prompt(
            self,
            prompts,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompts is not None and isinstance(prompts, str):
            batch_size = 1
        elif prompts is not None and isinstance(prompts, list):
            batch_size = len(prompts)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds_none_flag = (prompt_embeds is None)
        prompt_embeds_list = []
        embeds_pooler_list = []
        text_embeds_list = []
        for prompt in prompts:
            if prompt_embeds_none_flag:
                # textual inversion: procecss multi-vector tokens if necessary
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

                text_inputs = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                untruncated_ids = self.tokenizer(
                    prompt, padding="longest", return_tensors="pt"
                ).input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[
                    -1
                ] and not torch.equal(text_input_ids, untruncated_ids):
                    removed_text = self.tokenizer.batch_decode(
                        untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
                    )
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                    )

                if (
                        hasattr(self.text_encoder.config, "use_attention_mask")
                        and self.text_encoder.config.use_attention_mask
                ):
                    attention_mask = text_inputs.attention_mask.to(device)
                else:
                    attention_mask = None

                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device),
                    attention_mask=attention_mask,
                )
                embeds_pooler = prompt_embeds.pooler_output
                prompt_embeds = prompt_embeds[0]
                text_embeds = self.text_projector(text_input_ids.to(device)).text_embeds

            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
            embeds_pooler = embeds_pooler.to(dtype=self.text_encoder.dtype, device=device)
            text_embeds = text_embeds.to(dtype=self.text_encoder.dtype, device=device)

            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            embeds_pooler = embeds_pooler.repeat(1, num_images_per_prompt)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_images_per_prompt, seq_len, -1
            )
            embeds_pooler = embeds_pooler.view(
                bs_embed * num_images_per_prompt, -1
            )
            prompt_embeds_list.append(prompt_embeds)
            embeds_pooler_list.append(embeds_pooler)
            text_embeds_list.append(text_embeds)
        prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
        embeds_pooler = torch.cat(embeds_pooler_list, dim=0)
        text_embeds = torch.cat(text_embeds_list, dim=0)
        
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                negative_prompt = "worst quality, low quality, bad anatomy"
            uncond_tokens = [negative_prompt] * batch_size

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                    hasattr(self.text_encoder.config, "use_attention_mask")
                    and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            final_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return final_prompt_embeds, prompt_embeds, embeds_pooler[:, None, :], text_embeds

    @torch.no_grad()
    def __call__(
        self,
        control_image: torch.FloatTensor = None,
        prompt: List[List[str]] = None,
        obboxes: List[List[List[float]]] = None,
        bboxes: List[List[List[float]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        GUI_progress=None,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_nums = [0] * len(prompt)
        for i, _ in enumerate(prompt):
            prompt_nums[i] = len(_)

        device = self._execution_device
        self.image_encoder.to(device)
        self.text_projector.to(device)
        self.My_Proj_Model.to(device)
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, cond_prompt_embeds, embeds_pooler, text_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        ref_imgs = []
        for index, (caption, bndboxes) in enumerate(zip(prompt, bboxes)):
            categories = caption[1:]
            instances = []
            instance_imgs = []
            for name,bbox in zip(categories, bndboxes):
                if name == '':
                    instances.append(torch.zeros([3, 224, 224]))
                else:
                    value = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])    # foreground size based on HBB
                    chosen_file = list(dict_of_images[name].keys())[find_nearest(list(dict_of_images[name].values()), value)]                    
                    img = Image.open(os.path.join('path/to/foreground', name, chosen_file))   
                    instance_imgs.append(img)
                    img = self.feature_extractor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
                    instances.append(img)
            ref_imgs.append(torch.stack([instance for instance in instances]))
        
        ref_imgs = torch.stack([img for img in ref_imgs]).view(len(obboxes) * len(obboxes[0]), 3, 224, 224).to(device)
        condition_features = self.feature_extractor(images=control_image, return_tensors="pt")['pixel_values'].to(device)
        condition_features = self.image_encoder(condition_features)
        with torch.no_grad():
            img_features = self.image_encoder(ref_imgs)
            bg_img = get_similar_examplers(None, text_embeds[0], topk=1, sim_mode='text2img')
            bg_img = Image.open(os.path.join('path/to/train/image', bg_img[0])).convert('RGB')
            bg_features = self.feature_extractor(images=bg_img, return_tensors="pt")['pixel_values'].to(img_features.device)
            bg_features = self.image_encoder(bg_features)
        img_features, bg_features = self.My_Proj_Model(img_features, obboxes, bg_features, condition_features)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        controlnet_keep = []
        for i, t in enumerate(timesteps):
            controlnet_keep.append(1.0 if t < 850 else 0.0)  # 根据需求调整

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if GUI_progress is not None:
                    GUI_progress[0] = int((i + 1) / len(timesteps) * 100)
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                cross_attention_kwargs = {
                    'bboxes': bboxes,
                    'obboxes': obboxes,   
                    'embeds_pooler': embeds_pooler,
                    'height': height,
                    'width': width,
                    'ref_features': (img_features, bg_features),
                    'do_classifier_free_guidance': do_classifier_free_guidance,
                }
                if control_image is not None and controlnet_keep[i] > 0:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=control_image,
                        cross_attention_kwargs=cross_attention_kwargs,
                        conditioning_scale=1.0,
                        return_dict=False,
                    )
                else:
                    down_block_res_samples = None
                    mid_block_res_sample = None

                self.unet.eval()
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                step_output = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                latents = step_output.prev_sample

                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        
        if output_type == "latent":
            image = latents
        elif output_type == "pil":
            image = self.decode_latents(latents)
            image = self.numpy_to_pil(image)
        else:
            image = self.decode_latents(latents)
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)

if __name__ == '__main__':
    
    data_path = "path/to/data"
    ckpt_path = "path/to/checkpoint"
    clip_model_path = "path/to/clip"

    vae = AutoencoderKL.from_pretrained(ckpt_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_path)
    text_encoder = CLIPTextModel.from_pretrained(clip_model_path)
    unet = UNet2DConditionModel.from_pretrained(ckpt_path, subfolder="unet")
    scheduler = EulerDiscreteScheduler.from_pretrained(ckpt_path, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(clip_model_path)

    set_processors(unet)

    processor_ckpt_path = os.path.join(ckpt_path, 'unet', 'diffusion_pytorch_model.safetensors')
    if (os.path.exists(processor_ckpt_path)):
        state_dict = load_file(processor_ckpt_path)
        processor_state_dict = {k: v for k, v in state_dict.items() if ".processor" in k or ".self_attn" in k}
        attn_layers = AttnProcsLayers(unet.attn_processors)
        attn_layers.load_state_dict(processor_state_dict, strict=False)
    else:
        print("[Warning] No processor weight found at:", processor_ckpt_path)

    controlnet_weights_path = os.path.join(ckpt_path, "controlnet/diffusion_pytorch_model.safetensors")
    controlnet = CustomControlNetModel.from_pretrained(os.path.join(ckpt_path, 'controlnet'))
    set_processors(controlnet)
    state_dict = load_file(controlnet_weights_path, device="cpu")
    controlnet.load_state_dict(state_dict, strict=True)

    pipe = StableDiffusionMIPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
        safety_checker=None,
    )

    pipe.My_Proj_Model.load_state_dict(torch.load(os.path.join(ckpt_path, 'ImageProjModel.pth')))
    pipe.to("cuda")

    data = []
    with open(os.path.join(data_path, 'metadata.jsonl'), 'r') as f:
        for line in f:
            data.append(json.loads(line))

    negative_prompt = 'worst quality, low quality, bad anatomy, watermark, text, blurry'
    save_path_prefix = './gen_dior/2025_' + time.strftime('%m%d_%H%M', time.localtime(time.time()))
    
    for sample in data:
        file_name = sample['file_name']
        prompts = [sample['caption']]
        bboxes = [sample['bndboxes']]
        obboxes = [sample['obboxes']]
        condition_file_name = sample['condition_file_name']
        condition_image = Image.open(os.path.join(data_path, condition_file_name)).convert('RGB')
        conditioning_image_transforms = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
            ]
        )
        condition_image = conditioning_image_transforms(condition_image).to("cuda")
        condition_image = condition_image.unsqueeze(0)

        image = pipe(
            prompt=prompts,
            obboxes=obboxes,
            bboxes=bboxes,
            control_image=condition_image,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            height=512,
            width=512,
        ).images[0]

        if os.path.exists(save_path_prefix) == False:
            os.makedirs(save_path_prefix)
        image.save(os.path.join(save_path_prefix, file_name.split("/")[1]))
