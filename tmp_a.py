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
