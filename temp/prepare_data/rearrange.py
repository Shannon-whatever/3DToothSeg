import os
import shutil
import re
from tqdm import tqdm

# Symlink path (processed dataset)
dataset_root = "/home/zychen/Documents/Project_shno/3DToothSeg/temp/datasets/teeth3ds/processed"

# Split files path
split_root = "/home/zychen/Documents/Project_shno/3DToothSeg/dataset/teeth3ds/split"

# Output folder
output_root = "/home/zychen/Documents/Project_shno/3DToothSeg/temp/datasets/teeth3ds/teeth3ds_coco36"
os.makedirs(os.path.join(output_root, "train"), exist_ok=True)
os.makedirs(os.path.join(output_root, "test"), exist_ok=True)

def read_ids(file_path):
    """Read patient IDs from split file"""
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

# Collect train/test IDs
train_ids = []
train_ids += read_ids(os.path.join(split_root, "public-training-set-1.txt"))
train_ids += read_ids(os.path.join(split_root, "public-training-set-2.txt"))
test_ids = read_ids(os.path.join(split_root, "private-testing-set.txt"))

def select_images(images):
    """Select first 18 and last 18 images"""
    if len(images) <= 36:
        return images
    return images[:18] + images[-18:]

def extract_number(filename):
    name, _ = os.path.splitext(filename)
    match = re.findall(r"\d+", name)
    return int(match[-1]) if match else -1 

def copy_images(patient_id, split="train"):
    """Copy 36 render images for a given patient"""
    jaw = "upper" if "upper" in patient_id else "lower"
    patient_folder = os.path.join(dataset_root, jaw, patient_id.replace(f"_{jaw}", ""))
    render_folder = os.path.join(patient_folder, "render")

    if not os.path.exists(render_folder):
        print(f"⚠️ Missing render folder for {patient_id}")
        return 0

    images = sorted(
        [f for f in os.listdir(render_folder) if f.endswith(".png")],
        key=extract_number
    )
    selected = select_images(images)

    for img in selected:
        src = os.path.join(render_folder, img)
        dst = os.path.join(output_root, split, img)
        shutil.copy(src, dst)

    return len(selected)

# ---- Process all ----
print("🚀 Copying training images...")
for pid in tqdm(train_ids, desc="Training set"):
    copy_images(pid, "train")

print("🚀 Copying testing images...")
for pid in tqdm(test_ids, desc="Testing set"):
    copy_images(pid, "test")

print("✅ Finished organizing dataset into", output_root)
