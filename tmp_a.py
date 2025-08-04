# %%
import os
import trimesh
import numpy as np
from glob import glob
import json
import torch
import cv2
import matplotlib.pyplot as plt
from utils.other_utils import load_color_from_ply

num_points = 16000
file_path = '.datasets/teeth3ds/sample/processed/upper/YBSESUN6'
file_name = f'{file_path.split("/")[-1]}_{file_path.split("/")[-2]}'
mesh_path = os.path.join(file_path, f'{file_name}_process.ply')
mesh = trimesh.load(mesh_path)

point_coords = np.array(mesh.vertices)
face_info = np.array(mesh.faces)


cell_normals = np.array(mesh.face_normals)
cell_coords = np.array([
    [
        (point_coords[face[0]][0] + point_coords[face[1]][0] + point_coords[face[2]][0]) / 3,
        (point_coords[face[0]][1] + point_coords[face[1]][1] + point_coords[face[2]][1]) / 3,
        (point_coords[face[0]][2] + point_coords[face[1]][2] + point_coords[face[2]][2]) / 3,
    ]
    for face in face_info
])

pointcloud = np.concatenate((cell_coords, cell_normals), axis=1) # (N, 6)

if pointcloud.shape[0] < num_points:
    padding = np.zeros((num_points - pointcloud.shape[0], pointcloud.shape[1]))
    face_info = np.concatenate((face_info, np.zeros(shape=(num_points - pointcloud.shape[0], 3))), axis=0)
    pointcloud = np.concatenate((pointcloud, padding), axis=0)

# labels
labels = load_color_from_ply(mesh_path)

# %%

image_path_ls = sorted(glob(os.path.join(file_path, 'render', '*.png'))) 
label_path_ls = sorted(glob(os.path.join(file_path, 'mask', '*.png')))


image_path = image_path_ls[0]
label_path = label_path_ls[0]

with open(os.path.join(file_path, f'{file_name}_view.json'), 'r') as f:
    camera_params = json.load(f)

cameras_rt = camera_params[0]["Rt"]
cameras_k = camera_params[0]["K"]

# %%
render_size = (465, 465)
cameras_rt = torch.tensor(cameras_rt, dtype=torch.float32)[None, None, ...]  # (B, 4, 4)
cameras_k = torch.tensor(cameras_k, dtype=torch.float32)[None, None, ...]  # (B, 3, 3)
pointcloud = torch.tensor(pointcloud[:, :3], dtype=torch.float32)[None, ...]  # (1, N_pc, 3)

# %%
B, N_v, _, _ = cameras_rt.shape
_, N_pc, _ = pointcloud.shape

# 1. 变成齐次坐标 (B, N_pc, 4)
ones = torch.ones((B, N_pc, 1))
points_homo = torch.cat([pointcloud, ones], dim=-1)  # (B, N_pc, 4)

# 2. 外参变换：points_cam = Rt @ points_homo
Rt = cameras_rt[:, :, :3, :]
point_cam = torch.einsum('bvrc,bnc->bvnr', Rt, points_homo)  # (B, N_v, 3, 4) * (B, N_pc, 4) -> (B, N_v, N_pc, 3)

# 3. 内参投影：K @ points_cam
point_img = torch.einsum('bvrc,bvnc->bvnr', cameras_k, point_cam)  # (B, N_v, 3, 3) * (B, N_v, N_pc, 3) -> (B, N_v, N_pc, 3)

# 5. 归一化：除以z
img_x = point_img[..., 0] / (point_img[..., 2] + 1e-6) # x is w
img_y = point_img[..., 1] / (point_img[..., 2] + 1e-6) # y is h


img_points = torch.stack([img_y, render_size[0] - img_x], dim=-1)  # (B, N_v, N_pc, 2) 2: h, w
# img_points = torch.stack([img_y, img_x], dim=-1)

# %% 
image_cv = cv2.imread(image_path)          # 加载为 BGR 格式
image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)  # 转为 RGB 格式
img = (image_rgb * 255).astype(np.uint8) if image_rgb.max() <= 1 else image_rgb

# 获取点坐标 (N_pc, 2)
points = img_points[0, 0,...].numpy()
y, x = points[:, 0], points[:, 1]

# 可视化
plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.scatter(x, y)
plt.axis('off')
plt.show()



# %%
import os

def count_render_images(base_dir, expected_count=48):
    error_samples = []

    for subfolder in ['upper', 'lower']:
        sub_dir = os.path.join(base_dir, subfolder)
        for sample_id in os.listdir(sub_dir):
            sample_path = os.path.join(sub_dir, sample_id)
            render_dir = os.path.join(sample_path, 'render')

            if not os.path.isdir(render_dir):
                print(f"Warning: render folder not found for sample {sample_id} in {subfolder}")
                continue

            img_count = len([f for f in os.listdir(render_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if img_count != expected_count:
                error_samples.append((subfolder, sample_id, img_count))

    return error_samples

# 使用方式
base_path = '.datasets/teeth3ds/processed'  # 请根据实际路径修改
errors = count_render_images(base_path)

# 打印结果
if errors:
    print(f"\nSamples with image count not equal to 48:")
    for subfolder, sid, count in errors:
        print(f"{subfolder}/{sid}: {count} images")
else:
    print("All samples have 48 images.")

# 保存到 txt 文件
output_path = "exp/render_image_count_errors.txt"
with open(output_path, "w") as f:
    if errors:
        f.write("Samples with image count not equal to 48:\n")
        for subfolder, sid, count in errors:
            f.write(f"{subfolder}/{sid}: {count} images\n")
    else:
        f.write("All samples have 48 images.\n")

print(f"Results written to {output_path}")


# %%
import os
import matplotlib.pyplot as plt
from collections import defaultdict

base_dirs = ["./processed/upper", "./processed/lower"]
image_exts = {".png", ".jpg", ".jpeg"}

image_counts = {}

for base_dir in base_dirs:
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        continue

    for sample_id in os.listdir(base_dir):
        sample_path = os.path.join(base_dir, sample_id)
        render_dir = os.path.join(sample_path, "render")
        if not os.path.isdir(render_dir):
            print(f"No render dir for {sample_id}")
            continue

        images = [f for f in os.listdir(render_dir)
                  if os.path.isfile(os.path.join(render_dir, f)) and os.path.splitext(f)[-1].lower() in image_exts]

        image_counts[sample_id] = len(images)

# 输出每个 sample_id 的图片数量（供 debug）
for sid, count in image_counts.items():
    print(f"{sid}: {count} images")

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(list(image_counts.values()), bins=range(0, max(image_counts.values()) + 5, 2), color='skyblue', edgecolor='black')
plt.title("Distribution of Image Counts in Render Folders")
plt.xlabel("Number of Images")
plt.ylabel("Number of Sample IDs")
plt.grid(True)
plt.tight_layout()
plt.show()


# %%



# %%
import torchmetrics as tm
import torch


def calculate_miou(gt_labels, pred_labels, n_class=17, ignore_index=-1,
                   merge_pairs=[(1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15), (8, 16)]):
    """
    Args:
        pred_classes: Tensor (B, N), predicted class indices
        labels:       Tensor (B, N), ground truth class indices
        n_class:      int, number of classes

    Returns:
        miou_list: Tensor (B,), mIoU for each sample
        iou0_list: Tensor (B,), IoU for class 0 for each sample
    """
    device = gt_labels.device
    bs = gt_labels.shape[0]


    valid_mask = (gt_labels != ignore_index)

    pred_masked = pred_labels.clone()
    gt_masked = gt_labels.clone()

    pred_masked = pred_masked[valid_mask].reshape(bs, -1)
    gt_masked = gt_masked[valid_mask].reshape(bs, -1)
    
    cal_miou = tm.JaccardIndex(task="multiclass", num_classes=n_class).to(device)
    miou = cal_miou(pred_masked, gt_masked)

    cal_iou = tm.JaccardIndex(task="multiclass", num_classes=n_class, average=None).to(device)
    per_class_iou = cal_iou(pred_masked, gt_masked) # (C, )

    # 2. 计算合并类别IoU
    merged_ious = []
    for a, b in merge_pairs:
        gt_merge = (gt_labels == a) | (gt_labels == b)
        pred_merge = (pred_labels == a) | (pred_labels == b)

        # 二分类JaccardIndex计算 (B*num_pairs,)
        binary_iou_metric = tm.JaccardIndex(task="binary", num_classes=2).to(device)
        merged_iou = binary_iou_metric(pred_merge, gt_merge)
        merged_ious.append(merged_iou)

    merged_ious = torch.stack(merged_ious, dim=0)  # (num_pairs,)
    return miou, per_class_iou, merged_ious


def generate_random_labels(B=4, N=100, n_class=17, ignore_index=-1, ignore_prob=0.1):
    pred = torch.randint(low=0, high=n_class, size=(B, N))
    gt = torch.randint(low=0, high=n_class, size=(B, N))

    # 加一些 ignore_index
    mask = torch.rand((B, N)) < ignore_prob
    gt[mask] = ignore_index

    return gt, pred

# ------- 执行测试 -------
gt_labels, pred_labels = generate_random_labels(B=4, N=100, n_class=17, ignore_index=-1)

# 计算mIoU相关指标
miou, iou_per_class, merged_ious = calculate_miou(gt_labels, pred_labels, n_class=17, ignore_index=-1)
# %%
