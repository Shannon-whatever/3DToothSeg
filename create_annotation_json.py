import os
import cv2
from tqdm import tqdm

from .utils import FDI2color

def convert_teeth3ds_to_coco(root: str = '.datasets/teeth3ds', 
                             processed_folder: str = 'processed', 
                             mesh_view: list = ['upper', 'lower'], 
                             total_files: int = 1800,
                             is_train: bool = True,
                             train_test_split: int = 1,
                             output_path: str = '.datasets/teeth3ds'):
    categories = generate_categories_from_fdi(FDI2color)

    if train_test_split == 1:
        split_files = ['training_lower.txt', 'training_upper.txt'] if is_train else ['testing_lower.txt',
                                                                                         'testing_upper.txt']
    elif train_test_split == 2:
        split_files = ['public-training-set-1.txt', 'public-training-set-2.txt'] if is_train \
            else ['private-testing-set.txt']
    elif train_test_split == 0:
        split_files = ['training_lower_sample.txt', 'training_upper_sample.txt']
    else:
        raise ValueError(f'train_test_split should be 0, 1 or 2. not {train_test_split}')
    
    images = generate_images_from_teeth3ds(root, processed_folder, split_files, is_train)

    annotations = []

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }






def generate_categories_from_fdi(FDI2color):
    categories = []
    for fdi_id, (_, tooth_name, _) in FDI2color.items():
        categories.append({
            "id": fdi_id,          
            "name": tooth_name,    
            "supercategory": "tooth" if fdi_id != 0 else "gum"  
        })
    return categories

def generate_images_from_teeth3ds(root, processed_folder, split_files, is_train=True):
    images = []
    image_id = 1  

    for f in split_files: 
        with open(os.path.join(root, "split", f) , 'r') as file:
            for l in tqdm(file, desc=f"Processing {f}"):
                l_name, l_view = l.strip().split('_') 
                render_dir = os.path.join(root, processed_folder, l_view, l_name, "render")

                for img_file in sorted(os.listdir(render_dir)):
                    if img_file.endswith(".png"):  
                        img_path = os.path.join(render_dir, img_file)
                        img = cv2.imread(img_path)
                        height, width = img.shape[:2]

                        images.append({
                            "id": image_id,
                            "file_name": img_file,
                            "width": width,
                            "height": height
                        })

                        image_id += 1  # Increment image ID

    return images
        