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

from utils.other_utils import FDI2label


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
obj_file = '.datasets/teeth3ds/sample/upper/YBSESUN6/YBSESUN6_upper.obj'

mesh = trimesh.load(obj_file) # me3sh.faces shape (287970, 3)
with open(obj_file.replace('.obj', '.json')) as f:
    data = json.load(f)
labels = np.array(data["labels"]) # 144045
labels = labels[mesh.faces]
labels = labels[:, 0]
labels = np.array([FDI2label[label] for label in labels])
# labels_2 = fdi_to_label(labels)


# %%

mesh, labels = _donwscale_mesh(mesh, labels)

from collections import defaultdict, Counter

def face_labels_to_vertex_labels(faces, face_labels, num_vertices):
    # faces: (F, 3) int array of vertex indices per face
    # face_labels: (F,) int array of face labels
    # num_vertices: 顶点总数

    vertex_face_labels = defaultdict(list)

    for face_idx, face in enumerate(faces):
        label = face_labels[face_idx]
        for v in face:
            vertex_face_labels[v].append(label)

    vertex_labels = np.zeros(num_vertices, dtype=face_labels.dtype)
    for v in range(num_vertices):
        if v in vertex_face_labels:
            # 统计邻接面标签的众数
            c = Counter(vertex_face_labels[v])
            vertex_labels[v] = c.most_common(1)[0][0]
        else:
            vertex_labels[v] = -1  # 或者设置为无标签标识

    return vertex_labels

faces = mesh.faces
num_vertices = len(mesh.vertices)
vertex_labels = face_labels_to_vertex_labels(faces, labels, num_vertices)

# %%
from utils.other_utils import label2color_upper, output_pred_ply

# get gt mask

face_mask = []
for label in labels:
    color = label2color_upper[label][2]  # label 是单个 int
    face_mask.append(color)
face_mask = np.array(face_mask, dtype=np.uint8)  # shape: (N, 3)

vertex_mask = []
for label in vertex_labels:
    color = label2color_upper[label][2]  # label 是单个 int
    vertex_mask.append(color)
vertex_mask = np.array(vertex_mask, dtype=np.uint8)  # shape: (N, 3)


cell_normals = mesh.face_normals            # shape: (n_faces, 3)
point_coords = mesh.vertices                # shape: (n_vertices, 3)
face_info = mesh.faces                      # shape: (n_faces, 3)
output_pred_ply(face_mask, None, 'tmp/YBSESUN6_upper_mask.ply', point_coords, face_info, vertex_mask)




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
from utils.other_utils import color2label, load_color_from_ply

process_file = '.datasets/teeth3ds/sample/processed/upper/YBSESUN6_upper_process.ply'
num_points = 16000

mesh = trimesh.load(process_file)
            
cell_normals = np.array(mesh.face_normals)
point_coords = np.array(mesh.vertices)
face_info = np.array(mesh.faces)

# centroid of each face
cell_coords = np.array([
    [
        (point_coords[face[0]][0] + point_coords[face[1]][0] + point_coords[face[2]][0]) / 3,
        (point_coords[face[0]][1] + point_coords[face[1]][1] + point_coords[face[2]][1]) / 3,
        (point_coords[face[0]][2] + point_coords[face[1]][2] + point_coords[face[2]][2]) / 3,
    ]
    for face in face_info
])

pointcloud = np.concatenate((cell_coords, cell_normals), axis=1)

if pointcloud.shape[0] < num_points:
    padding = np.zeros((num_points - pointcloud.shape[0], pointcloud.shape[1]))
    face_info = np.concatenate((face_info, np.zeros(shape=(num_points - pointcloud.shape[0], 3))), axis=0)
    pointcloud = np.concatenate((pointcloud, padding), axis=0)



# labels
labels = load_color_from_ply(process_file)
labels = torch.from_numpy(labels)


# %%
permute = np.random.permutation(num_points)
pointcloud = pointcloud[permute]
face_info = face_info[permute]
labels = labels[permute]


# %%

def load_color_from_ply(file_path):
    plydata = PlyData.read(file_path)
    face = plydata['face']
    # 获取面颜色
    if 'red' in face and 'green' in face and 'blue' in face:
        colors = np.stack([face['red'], face['green'], face['blue']], axis=1)
    else:
        raise ValueError("No face color info in PLY file")

    labels = []
    for c in colors:
        c_tuple = tuple(c)
        assert c_tuple in color2label, f"Color {c_tuple} not found in color2label"

        labels.append(color2label[c_tuple][2])

    return np.array(labels, dtype=np.int64)


# %%
from plyfile import PlyData

plydata = PlyData.read(process_file)
face = plydata['face']

# 获取所有顶点坐标

# 获取面顶点索引
face_indices = face['vertex_indices']
# 获取面颜色
if 'red' in face and 'green' in face and 'blue' in face:
    colors = np.stack([face['red'], face['green'], face['blue']], axis=1)
else:
    raise ValueError("PLY文件的face没有颜色属性")


face_xyzs = []
for idxs in face_info:
    idxs = list(idxs)
    # 按xyz排序顶点
    pts = point_coords[idxs]
    pts_sorted = pts[np.lexsort((pts[:,2], pts[:,1], pts[:,0]))]
    face_xyzs.append(pts_sorted.flatten())




# np.array(labels, dtype=np.int32), np.array(face_xyzs)
# %%
import numpy as np
from plyfile import PlyData
import trimesh
import pyrender
from utils.other_utils import color2label

# 创建模型
model_path = '.datasets/teeth3ds/sample/processed/upper/YBSESUN6_upper_process.ply'
label_trimesh = trimesh.load(model_path)
pyrender_mesh = pyrender.Mesh.from_trimesh(label_trimesh)

# 创建场景
scene = pyrender.Scene()
label_scene = pyrender.Scene()

# 场景添加模型
scene.add(pyrender_mesh)
seg_node_map = {}
vertex_instances = {}
label_color_map = {}
face_instances = {}
for i, vertex_color in enumerate(label_trimesh.visual.vertex_colors):
    vertex_color = (vertex_color[0], vertex_color[1], vertex_color[2])
    if vertex_color in color2label:
        vertex_label = color2label[vertex_color][2]
        if not vertex_label in vertex_instances:
            vertex_instances[vertex_label] = {}
        vertex_instances[vertex_label][i] = len(vertex_instances[vertex_label])
        label_color_map[vertex_label] = vertex_color
for i, face in enumerate(label_trimesh.faces):
    for label, vertices in vertex_instances.items():
        if face[0] in vertices and face[1] in vertices and face[2] in vertices:
            if not label in face_instances:
                face_instances[label] = []
            face_instances[label].append([vertices[face[0]], vertices[face[1]], vertices[face[2]]])
for label, vertices in vertex_instances.items():
    label_color = label_color_map[label]
    label_color = [label_color[0], label_color[1], label_color[2]]
    vertice_node = np.array([label_trimesh.vertices[i] for i, _ in vertices.items()], dtype=float)
    vertice_color_node = np.array([label_color] * vertice_node.shape[0])
    face_node = np.array(face_instances[label])
    face_color_node = np.array([label_color] * face_node.shape[0])
    mesh_node = trimesh.Trimesh(vertices=vertice_node, faces=face_node, vertex_colors=vertice_color_node, face_colors=face_color_node)

    # 当前模型添加到场景中
    node = label_scene.add(pyrender.Mesh.from_trimesh(mesh_node))
    seg_node_map[node] = label_color
# %%
