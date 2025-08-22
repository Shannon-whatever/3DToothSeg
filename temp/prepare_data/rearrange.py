import os
import shutil
import re
from tqdm import tqdm
import argparse

# -------------------------------
# Helper functions
# -------------------------------
def read_ids(file_path):
    """Read patient IDs from split file"""
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

def select_images(images, n=18):
    """Select first n and last n images"""
    if len(images) <= 2 * n:
        return images
    return images[:n] + images[-n:]

def extract_number(filename):
    """Extract last number from filename for sorting"""
    name, _ = os.path.splitext(filename)
    match = re.findall(r"\d+", name)
    return int(match[-1]) if match else -1 

def copy_images(patient_id, dataset_root, output_root, split="train", is_mask=True):
    """Copy images (mask or render) for a given patient"""
    jaw = "upper" if "upper" in patient_id else "lower"
    patient_folder = os.path.join(dataset_root, jaw, patient_id.replace(f"_{jaw}", ""))

    folder_type = "mask" if is_mask else "render"
    src_folder = os.path.join(patient_folder, folder_type)

    if not os.path.exists(src_folder):
        print(f"⚠️ Missing {folder_type} folder for {patient_id}")
        return 0

    images = sorted(
        [f for f in os.listdir(src_folder) if f.endswith(".png")],
        key=extract_number
    )
    # selected = select_images(images)

    # Make sure destination folder exists
    dst_folder = os.path.join(output_root, f"{split}_{folder_type}")
    os.makedirs(dst_folder, exist_ok=True)

    for img in images:
        src = os.path.join(src_folder, img)
        dst = os.path.join(dst_folder, img)
        shutil.copy(src, dst)

    return len(images)

# -------------------------------
# Main function
# -------------------------------
def main(args):
    # Collect train/test IDs
    train_ids = read_ids(os.path.join(args.split_root, "train.txt"))
    test_ids = read_ids(os.path.join(args.split_root, "test.txt"))

    # Process training set
    print("🚀 Copying training images...")
    for pid in tqdm(train_ids, desc="Training set"):
        copy_images(pid, args.dataset_root, args.output_root, split="train", is_mask=args.is_mask)

    # Process testing set
    print("🚀 Copying testing images...")
    for pid in tqdm(test_ids, desc="Testing set"):
        copy_images(pid, args.dataset_root, args.output_root, split="test", is_mask=args.is_mask)

    print("✅ Finished organizing dataset into", args.output_root)

# -------------------------------
# Argument parser
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize teeth3ds dataset")
    parser.add_argument("--dataset_root", type=str, default="/home/zychen/Documents/Project_shno/3DToothSeg/temp/datasets/teeth3ds/processed")
    parser.add_argument("--split_root", type=str, default="/home/zychen/Documents/Project_shno/3DToothSeg/temp/datasets/teeth3ds/split")
    parser.add_argument("--output_root", type=str, default="/home/zychen/Documents/Project_shno/3DToothSeg/temp/datasets/teeth3ds/teeth3ds_coco")
    parser.add_argument("--is_mask", action="store_true", help="Copy mask files if set; otherwise copy render files")
    args = parser.parse_args()

    main(args)