"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import random
import sys

import cv2  # Added for video processing
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig

label_map = {
    1: 'gametocyte', 2: 'ring', 3:'schizont', 4:'trophozoite'
}


COLORS = plt.cm.tab20.colors  
COLOR_MAP = {label: tuple([int(c * 255) for c in COLORS[i % len(COLORS)]]) for i, label in enumerate(label_map)}



def draw(image, labels, boxes, scores, thrh=0.45):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default() 
    labels, boxes, scores = labels[scores > thrh], boxes[scores > thrh], scores[scores > thrh]

    for j, box in enumerate(boxes):
        category = labels[j].item()
        color = COLOR_MAP.get(category, (255, 255, 255))  
        box = list(map(int, box))

        
        draw.rectangle(box, outline=color, width=3)

        text = f"{label_map[category]} {scores[j].item():.2f}"
        text_bbox = draw.textbbox((0, 0), text, font=font)  
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        
        text_background = [box[0], box[1] - text_height - 2, box[0] + text_width + 4, box[1]]
        draw.rectangle(text_background, fill=color)
       
        draw.text((box[0] + 2, box[1] - text_height - 2), text, fill="black", font=font)

    return image


def process_dataset(model, dataset_path, output_path, thrh=0.5, size=(640, 640), vit_backbone=False):
    os.makedirs(output_path, exist_ok=True)
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png', '.JPG'))]

    transforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
                if vit_backbone else T.Lambda(lambda x: x)
    ])

    print(f"Found {len(image_paths)} images in validation set...")
    for idx, file_path in enumerate(image_paths):
        im_pil = Image.open(file_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).cuda()

        # 图像预处理
        im_data = transforms(im_pil).unsqueeze(0).cuda()
        output = model(im_data, orig_size)
        labels, boxes, scores = output[0]['labels'], output[0]['boxes'], output[0]['scores']

        # 绘制结果
        vis_image = draw(im_pil.copy(), labels, boxes, scores, thrh)
        save_path = os.path.join(output_path, f"vis_{os.path.basename(file_path)}")
        vis_image.save(save_path)

        if idx % 500 == 0:
            print(f"Processed {idx}/{len(image_paths)} images...")

    print("Visualization complete. Results saved in:", output_path)


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.eval().cuda()
            self.postprocessor = cfg.postprocessor.eval().cuda()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()
    img_size = cfg.yaml_cfg["eval_spatial_size"]
    vit_backbone = cfg.yaml_cfg.get('DINOv3STAs', False)

    process_dataset(model, args.dataset, args.output, thrh=0.45, size=img_size, vit_backbone=vit_backbone)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-d', '--dataset', type=str, default='./data/fiftyone/validation/data')
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to save visualized results")
    args = parser.parse_args()
    main(args)
