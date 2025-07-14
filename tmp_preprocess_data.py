# %%
import json
import os
import pickle
import trimesh
import numpy as np
import pyfqmr
from scipy import spatial
import torch

from utils.teeth_numbering import fdi_to_label


# %%
def _donwscale_mesh(mesh, labels):
        mesh_simplifier = pyfqmr.Simplify()
        mesh_simplifier.setMesh(mesh.vertices, mesh.faces)
        mesh_simplifier.simplify_mesh(target_count=16000, aggressiveness=3, preserve_border=True, verbose=0,
                                      max_iterations=2000)
        new_positions, new_face, _ = mesh_simplifier.getMesh()
        mesh_simple = trimesh.Trimesh(vertices=new_positions, faces=new_face)
        vertices = mesh_simple.vertices
        faces = mesh_simple.faces
        if faces.shape[0] < 16000:
            fs_diff = 16000 - faces.shape[0]
            faces = np.append(faces, np.zeros((fs_diff, 3), dtype="int"), 0)
        elif faces.shape[0] > 16000:
            mesh_simple = trimesh.Trimesh(vertices=vertices, faces=faces)
            samples, face_index = trimesh.sample.sample_surface_even(mesh_simple, 16000)
            mesh_simple = trimesh.Trimesh(vertices=mesh_simple.vertices, faces=mesh_simple.faces[face_index])
            faces = mesh_simple.faces
            vertices = mesh_simple.vertices
        mesh_simple = trimesh.Trimesh(vertices=vertices, faces=faces)

        mesh_v_mean = mesh.vertices[mesh.faces].mean(axis=1)
        mesh_simple_v = mesh_simple.vertices
        tree = spatial.KDTree(mesh_v_mean)
        query = mesh_simple_v[faces].mean(axis=1)
        distance, index = tree.query(query)
        labels = labels[index].flatten()
        return mesh_simple, labels

# %%
obj_file = '.datasets/teeth3ds/sample/YBSESUN6_upper.obj'

mesh = trimesh.load(obj_file) # me3sh.faces shape (287970, 3)
with open(obj_file.replace('.obj', '.json')) as f:
    data = json.load(f)
labels = np.array(data["labels"]) # 144045
labels = labels[mesh.faces]
labels = labels[:, 0]
labels = fdi_to_label(labels)

# %%

mesh, labels = _donwscale_mesh(mesh, labels)

# %%
from utils.other_utils import label2color_upper, output_pred_ply

# get gt mask

mask = []
for label in labels:
    color = label2color_upper[label][2]  # label 是单个 int
    mask.append(color)
mask = np.array(mask, dtype=np.uint8)  # shape: (N, 3)

cell_normals = mesh.face_normals            # shape: (n_faces, 3)
point_coords = mesh.vertices                # shape: (n_vertices, 3)
face_info = mesh.faces                      # shape: (n_faces, 3)
output_pred_ply(mask, None, 'tmp/YBSESUN6_upper_mask.ply', point_coords, face_info)




# %%
from utils.other_utils import FDI2color, color2label
def build_fdi2label(FDI2color, color2label):
    fdi2label = {}
    for fdi, (_, _, rgb) in FDI2color.items():
        if rgb in color2label:
            fdi2label[fdi] = color2label[rgb][2]  # 第三个元素是 label index
        else:
            print(f"Warning: RGB {rgb} for FDI {fdi} not found in color2label!")
    return fdi2label

FDI2label_map = build_fdi2label(FDI2color, color2label)
print(FDI2label_map)


# %%
