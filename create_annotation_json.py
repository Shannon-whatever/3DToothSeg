import os
import torch
import json
from pathlib import Path
import PIL as Image
import numpy as np
from torchvision.ops import masks_to_boxes
from tqdm import tqdm


from .utils import FDI2color, color2label

def create_annotation_json(root: str = '/home/zychen/Documents/Project_shno/3DToothSeg/dataset/teeth3ds/teeth3ds',
                           split_folder: str = 'split',
                           processed_folder: str = 'processed',
                           txt_file: str = 'training_lower_sample.txt',
                           output_dir: str = '/home/zychen/Documents/Project_shno/3DToothSeg/dataset/teeth3ds/teeth3ds/annotation'):
    
    categories = []
    for fdi_id, (_, tooth_name, _) in FDI2color.items():
        categories.append({
            "id": fdi_id,          
            "name": tooth_name,    
            "supercategory": "tooth" if fdi_id != 0 else "gum"  
        })

    images = []
    annotations = []
    annotation_id = 1

    with open(os.path.join(root, split_folder, txt_file), 'r') as file:
        for line in tqdm(file, desc="Processing Patients"):
            line = line.rstrip()
            l_name = line.split('_')[0]
            l_view = line.split('_')[0]

            render_dir = os.path.join(root, 'sample', processed_folder, l_view, l_name, 'render')
            mask_dir = os.path.join(root, 'sample', processed_folder, l_view, l_name, 'mask')

            if not os.path.exists(render_dir) or not os.path.exists(mask_dir):
                print(f"Warning: Missing directories for {line}")
                continue

            render_files = sorted(os.listdir(render_dir))
            mask_files = sorted(os.listdir(mask_dir))

            for render_file, mask_file in zip(render_files, mask_files):
                render_path = os.path.join(render_dir, render_file)
                mask_path = os.path.join(mask_dir, mask_file)

                img = Image.open(render_path).convert("RGB")
                width, height = img.size
                image_id = len(images) + 1
                images.append({
                    "id": image_id,
                    "file_name": render_files,
                    "width": width,
                    "height": height
                })

                mask = np.array(Image.open(mask_path).convert("RGB"))
                unique_colors = np.unique(mask.reshape(-1, 3), axis=0)
                instance_colors = [tuple(color) for color in unique_colors if tuple(color) in color2label]

                for color in instance_colors:
                    if color not in color2label:
                        print(f"Warning: Color {color} not found in color2label. Skipping.")
                        continue
                    binary_mask = (np.all(mask == np.array(color), axis=-1)).astype(np.uint8)
                    binary_mask_tensor = torch.tensor(binary_mask, dtype=torch.uint8).unsqueeze(0)
                    if binary_mask_tensor.sum() == 0:  
                        continue
                    
                    box = masks_to_boxes(binary_mask_tensor).to(torch.float32)
                    box[:, 2:] -= box[:, :2]  # Convert to (x_min, y_min, w, h)
                    box[:, 0::2].clamp_(min=0, max=width)
                    box[:, 1::2].clamp_(min=0, max=height)

                    category_id = color2label[color][2]

                    for b in box:
                        annotations.append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": [int(coord) for coord in b.tolist()],
                            "area": int(b[2] * b[3]),  # Area = width * height
                            "iscrowd": 0
                        })
                        annotation_id += 1

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
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    output_file_name = f"{os.path.splitext(os.path.basename(txt_file))[0]}_annotation.json"
    output_path = os.path.join(output_dir, output_file_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as json_file:
        json.dump(coco_format, json_file, indent=4)

    print(f"Annotation JSON saved at {output_path}")