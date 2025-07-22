# %%
import os
import trimesh
import numpy as np
import json
from utils.other_utils import FDI2label, label2color_upper, label2color_lower
from utils.other_utils import face_labels_to_vertex_labels, output_pred_ply

# change these paths to your own
file = '.datasets/teeth3ds/sample/upper/YBSESUN6/YBSESUN6_upper.obj' # data path
save_path = 'tmp/YBSESUN6_upper_gt.ply' # save path

mesh = trimesh.load(file)

with open(file.replace('.obj', '.json')) as f:
    data = json.load(f)
labels = np.array(data["labels"])
labels = labels[mesh.faces]
labels = labels[:, 0]
labels = np.array([FDI2label[label] for label in labels])


mask = []
for label in labels:
    if 'upper' in file:
        color = label2color_upper[label][2]  # label 是单个 int
    elif 'lower' in file:
        color = label2color_lower[label][2]
    mask.append(color)
mask = np.array(mask, dtype=np.uint8)  # shape: (N, 3)

# get vertex mask              # shape: (n_vertices, 3)

vertex_labels = face_labels_to_vertex_labels(mesh.faces, labels, len(mesh.vertices))
vertex_mask = []
for label in vertex_labels:
    if 'upper' in file:
        color = label2color_upper[label][2]  # label 是单个 int
    elif 'lower' in file:
        color = label2color_lower[label][2]
    vertex_mask.append(color)
vertex_mask = np.array(vertex_mask, dtype=np.uint8)  # shape: (N, 3)

point_coords = mesh.vertices                # shape: (n_vertices, 3)
face_info = mesh.faces                      # shape: (n_faces, 3)
output_pred_ply(mask, None, save_path, point_coords, face_info, vertex_mask)


