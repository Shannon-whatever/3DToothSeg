import os
import re
import torch
import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import time
import argparse
from torchvision.ops import masks_to_boxes, box_convert
from utils.color_utils import label2color_lower, color2label

def extract_number(filename):
    """Extract last integer from filename for numeric sorting"""
    name, _ = os.path.splitext(filename)
    match = re.findall(r"\d+", name)
    return int(match[-1]) if match else -1

def create_annotation_json(root, txt_file, output_dir):
    categories = []
    for seq_id, (_, tooth_name, _) in label2color_lower.items():
        # Skip gum, category_id starts from 1
        if seq_id == 0:
            continue
        categories.append({
            "id": seq_id,
            "name": tooth_name[-2:],
            "supercategory": "tooth" if seq_id != 0 else "gum"
        })

    images = []
    annotations = []
    image_id = 1
    annotation_id = 1

    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"Split file {txt_file} does not exist.")
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        for line in tqdm(lines, desc="Processing Patients", total=len(lines)):
            # patient_start_time = time.time()
            line = line.rstrip()
            l_name = line.split('_')[0]
            l_view = line.split('_')[1]

            render_dir = os.path.join(root, l_view, l_name, 'render')
            mask_dir = os.path.join(root, l_view, l_name, 'mask')
            if not os.path.exists(render_dir) or not os.path.exists(mask_dir):
                tqdm.write(f"Warning: Missing directories for {line}")
                continue

            render_files = sorted(os.listdir(render_dir), key=extract_number)
            mask_files = sorted(os.listdir(mask_dir), key=extract_number)

            # Select first 18 and last 18 (if enough exist)
            if len(render_files) > 36:
                render_files = render_files[:18] + render_files[-18:]
                mask_files = mask_files[:18] + mask_files[-18:]

            for render_file, mask_file in list(zip(render_files, mask_files)):
                render_path = os.path.join(render_dir, render_file)
                mask_path = os.path.join(mask_dir, mask_file)

                img = Image.open(render_path).convert("RGB")
                width, height = img.size
                images.append({
                    "id": image_id,
                    "file_name": render_file,
                    "width": width,
                    "height": height
                })

                mask = torch.from_numpy(np.array(Image.open(mask_path).convert("RGB"))).cuda()
                colors_tensor = torch.tensor([color for color, (_, _, cat_id) in color2label.items() if cat_id != 0], dtype=torch.uint8).cuda()
                mask_reshaped = mask.reshape(-1, 3)
                colors_reshaped = colors_tensor.reshape(-1, 1, 3)

                binary_masks = (mask_reshaped == colors_reshaped).all(dim=2)  # (N, H*W)
                binary_masks = binary_masks.reshape(len(colors_tensor), mask.shape[0], mask.shape[1])

                for i, binary_mask in enumerate(binary_masks):
                    if not binary_mask.any():  
                        continue
                    color = tuple(colors_tensor[i].cpu().numpy())
                    (_, _ , category_id) = color2label[color]

                    binary_mask = binary_mask.unsqueeze(0)
                    box = masks_to_boxes(binary_mask).to(torch.float32)

                    box_wh = box_convert(box, in_fmt="xyxy", out_fmt="xywh")
                    box_wh[:, 0::2].clamp_(min=0, max=width)
                    box_wh[:, 1::2].clamp_(min=0, max=height)

                    for b in box_wh:
                        annotations.append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": [int(coord) for coord in b.tolist()],
                            "bbox_mode": 1, # BoxMode.XYWH_ABS
                            "iscrowd": 0
                        })
                        annotation_id += 1
                
                image_id += 1
            # patient_end_time = time.time()
            # print(f"Processed patient in {patient_end_time - patient_start_time:.2f} seconds")

    coco_format = {
        "info": {
            "description": "3D Tooth Segmentation Dataset",
            "url": "https://github.com/Shannon-whatever/3DToothSeg",
            "version": "1.0",
            "year": 2025,
            "contributor": "Chensy",
            "date_created": "2025-08-07"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)",
                "url": "https://creativecommons.org/licenses/by-nc/4.0/"
            }
        ],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }

    output_file_name = f"{os.path.splitext(os.path.basename(txt_file))[0]}_annotation.json"
    output_path = os.path.join(output_dir, output_file_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as json_file:
        json.dump(coco_format, json_file, indent=4)

    print(f"Annotation JSON saved at {output_path}")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_file', type=str, required=True, help='Path to the text file containing image paths')
    parser.add_argument('--output_dir', type=str, default='/home/zychen/Documents/Project_shno/3DToothSeg/temp/datasets/teeth3ds/teeth3ds_coco36')
    args = parser.parse_args()

    create_annotation_json(
        root='/home/zychen/Documents/Project_shno/3DToothSeg/dataset/teeth3ds/processed',
        txt_file=args.txt_file,
        output_dir=args.output_dir
    )