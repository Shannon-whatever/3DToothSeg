import os
import torch
import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
from torchvision.ops import masks_to_boxes, box_convert
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

from utils.color_utils import FDI2color, color2label, color2FDI, fdi_to_sequential_id

def create_annotation_json(root: str = '/home/zychen/Documents/Project_shno/3DToothSeg/dataset/teeth3ds/teeth3ds',
                           split_folder: str = 'split',
                           processed_folder: str = 'processed',
                           is_train: bool = True,
                           train_test_split: int = 0,
                           image_set: str = 'train',
                           output_dir: str = '/home/zychen/Documents/Project_shno/3DToothSeg/dataset/teeth3ds/teeth3ds/annotation'):
    
    if train_test_split == 1:
        split_files = ['training_lower.txt', 'training_upper.txt'] if is_train else ['testing_lower.txt',                                                                           'testing_upper.txt']
    elif train_test_split == 2:
        split_files = ['public-training-set-1.txt', 'public-training-set-2.txt'] if is_train \
            else ['private-testing-set.txt']
    elif train_test_split == 0:
        if image_set == 'train':
            split_files = ['training_lower_sample.txt', 'training_upper_sample.txt']
        elif image_set == 'val':
            split_files = ['validation_lower_sample.txt', 'validation_upper_sample.txt']
        elif image_set == 'test':
            split_files = ['testing_lower_sample.txt', 'testing_upper_sample.txt']
    else:
        raise ValueError(f'train_test_split should be 0, 1 or 2. not {train_test_split}')
    
    for split_file in split_files:
        txt_file = os.path.join(root, split_folder, split_file)
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"Split file {txt_file} does not exist.")

        categories = []
        for fdi_id, (_, tooth_name, _) in FDI2color.items():
            if fdi_id == 0:
                continue
            categories.append({
                "id": fdi_to_sequential_id[fdi_id],          
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
                l_view = line.split('_')[1]

                render_dir = os.path.join(root, 'sample', processed_folder, l_view, l_name, 'render')
                mask_dir = os.path.join(root, 'sample', processed_folder, l_view, l_name, 'mask')
                bbox_dir = os.path.join(root, 'sample', processed_folder, l_view, l_name, 'bbox')

                Path(bbox_dir).mkdir(parents=True, exist_ok=True)
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
                        "file_name": render_file,
                        "width": width,
                        "height": height
                    })

                    mask = np.array(Image.open(mask_path).convert("RGB"))
                    unique_colors = np.unique(mask.reshape(-1, 3), axis=0)
                    instance_colors = [tuple(color) for color in unique_colors if tuple(color) in color2label]

                    boxes = []
                    colors = []
                    labels = []

                    for color in instance_colors:
                        if color not in color2label:
                            print(f"Warning: Color {color} not found in color2label. Skipping.")
                            continue

                        fdi_id = color2FDI[color]    
                        if fdi_id == 0:
                            continue
                        category_id = fdi_to_sequential_id[fdi_id]

                        binary_mask = (np.all(mask == np.array(color), axis=-1)).astype(np.uint8)
                        binary_mask_tensor = torch.tensor(binary_mask, dtype=torch.uint8).unsqueeze(0)
                        if binary_mask_tensor.sum() == 0:
                            print(f"Skipping empty mask for color {color}.")
                            continue
                        
                        box = masks_to_boxes(binary_mask_tensor).to(torch.float32)
                        boxes.append(box)
                        colors.append(tuple(color))
                        labels.append(color2label[color][1])

                        box_wh = box_convert(box, in_fmt="xyxy", out_fmt="xywh")
                        box_wh[:, 0::2].clamp_(min=0, max=width)
                        box_wh[:, 1::2].clamp_(min=0, max=height)

                        for b in box_wh:
                            annotations.append({
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": category_id,
                                "bbox": [int(coord) for coord in b.tolist()],
                                "area": int(b[2] * b[3]),  # Area = width * height
                                "iscrowd": 0
                            })
                            annotation_id += 1

                    if boxes:
                        boxes_tensor = torch.cat(boxes, dim=0)
                        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1) 
                        img_with_boxes = draw_bounding_boxes(
                            img_tensor,
                            boxes=boxes_tensor,
                            labels=labels,
                            colors=[f"rgb({r},{g},{b})" for r, g, b in colors]
                        )
                        img_with_boxes_pil = to_pil_image(img_with_boxes)
                        img_with_boxes_pil.save(os.path.join(bbox_dir, render_file))
                    else:
                        print(f"No bounding boxes found for {render_file}. Skipping visualization.")
                        continue
            

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an annotation JSON file.")
    parser.add_argument("--is_train", type=bool, default=True)
    parser.add_argument("--image_set", type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument("--train_test_split", type=int, default=0, choices =[0, 1, 2])
    args = parser.parse_args()

    create_annotation_json(is_train=args.is_train, train_test_split=args.train_test_split, image_set=args.image_set)