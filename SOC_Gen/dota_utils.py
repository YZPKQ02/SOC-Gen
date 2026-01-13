import functools
import os
import random
import numpy as np
import imagesize
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from shapely.geometry import Polygon


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

ref_img_path = 'path/to/dota/results/foreground'
dict_of_images = {}
list_of_name =("plane","ship","storage-tank","baseball-diamond","tennis-court",
                "basketball-court","ground-track-field","harbor","bridge","large-vehicle",
                "small-vehicle","helicopter","roundabout","soccer-ball-field","swimming-pool")
for name in list_of_name:
    name_of_dir = os.path.join(ref_img_path, name)
    list_of_image = os.listdir(name_of_dir)
    list_of_image = sorted(list_of_image, 
                           key = lambda img: functools.reduce(lambda x, y: x*y, imagesize.get(os.path.join(name_of_dir, img))), 
                           reverse=True)
    dict_of_images[name] = {img: functools.reduce(lambda x, y: x/y, imagesize.get(os.path.join(name_of_dir, img))) 
                            for img in list_of_image[:200]}

def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array/value - 1)).argmin()
        return idx

def get_sup_mask(mask_list):
    or_mask = np.zeros_like(mask_list[0])
    for mask in mask_list:
        or_mask += mask
    or_mask[or_mask >= 1] = 1
    sup_mask = 1 - or_mask
    return sup_mask

def composite_images(base_image_size, ref_imgs, positions):
    base_image = Image.new('RGB', base_image_size, (0, 0, 0))
    W, H = base_image_size
    for ref_img, (x1, y1, x2, y2) in zip(ref_imgs, positions):
        w = x2 - x1
        h = y2 - y1
        if w == 0 or h == 0:
            continue
        x, y, w, h = x1 * W, y1 * H, w * W, h * H
        img_resized = ref_img.resize((int(w), int(h)))
        
        base_image.paste(img_resized, (int(x), int(y)))

    return base_image    
    

data_emb_dict = torch.load('path/to/dota_emb_train.pt')
    
def get_similar_examplers(query_img_name, prompt_emb, topk=5, sim_mode='both'):
    prompt_emb = F.normalize(prompt_emb, dim=-1).detach().cpu()

    img_name_list = []
    sim_val_list = []
    for img_name, data_emb in data_emb_dict.items():
        img_name_list.append(img_name)
        txt_emb = data_emb['txt_emb']
        img_emb = data_emb['img_emb']

        if sim_mode == 'text2text':
            sim_val = (prompt_emb * txt_emb).sum(dim=-1)
        elif sim_mode == 'text2img':
            sim_val = (prompt_emb * img_emb).sum(dim=-1)
        elif sim_mode == 'both':
            txt_sim_val = (prompt_emb * txt_emb).sum(dim=-1)
            img_sim_val = (prompt_emb * img_emb).sum(dim=-1)
            sim_val = (txt_sim_val + img_sim_val) * 0.5
        else:
            raise ValueError('Invalid mode for similarity computation! (text2text | text2img | both)')
    
        sim_val_list.append(sim_val.item())

    sim_val_list, img_name_list = zip(*sorted(zip(sim_val_list, img_name_list)))
    sim_val_list = list(sim_val_list)
    img_name_list = list(img_name_list)

    img_emb_list = [data_emb_dict[img_name]['img_emb'] for img_name in img_name_list[-topk:]]

    return img_name_list[-topk:]

def draw_adaptive_polygon_mask(obboxes, image_size):
    H, W = image_size
    total_area = H * W
    weight_mask = np.ones((H, W), dtype=np.float32)

    for obb in obboxes:
        if isinstance(obb, torch.Tensor):
            obb = obb.tolist()
        if len(obb) != 8:
            continue

        poly = Polygon([
            (obb[0], obb[1]), (obb[2], obb[3]),
            (obb[4], obb[5]), (obb[6], obb[7])
        ])
        area = poly.area
        if area <= 1e-5:
            continue

        weight = np.clip(total_area / area, 1.0, 50.0)

        mask_img = Image.new('L', (W, H), 0)
        draw = ImageDraw.Draw(mask_img)
        draw.polygon(obb, fill=1)
        mask = np.array(mask_img, dtype=np.float32)

        weight_mask = np.where(mask > 0, np.maximum(weight_mask, weight), weight_mask)

    return torch.from_numpy(weight_mask).float()

def generate_adaptive_weight_masks(batch_obboxes, image_size=(512, 512), target_size=(64, 64)):
    H, W = image_size
    h, w = target_size
    B = len(batch_obboxes)
    masks = []

    for i in range(B):
        weight_mask = draw_adaptive_polygon_mask(batch_obboxes[i], image_size)
        weight_mask = F.interpolate(
            weight_mask.unsqueeze(0).unsqueeze(0),
            size=target_size,
            mode='area'
        )
        masks.append(weight_mask)

    return torch.cat(masks, dim=0)