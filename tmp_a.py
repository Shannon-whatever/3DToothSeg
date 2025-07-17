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
    else:
        print(f"Vertex color {vertex_color} not found in color2label mapping.")

face_instances = {}
if 'face' in label_trimesh.metadata['_ply_raw']:
    face_meta = label_trimesh.metadata['_ply_raw']['face']
    if 'red' in face_meta['data'] and 'green' in face_meta['data'] and 'blue' in face_meta['data']:
        face_colors = np.stack([
            face_meta['data']['red'],
            face_meta['data']['green'],
            face_meta['data']['blue']
        ], axis=-1).squeeze(1)  # shape: (N_faces, 3)

# %%  
for i, face in enumerate(label_trimesh.faces):
    face_color = tuple(face_colors[i])  # 取RGB，忽略alpha

    if face_color in color2label:
        label = color2label[face_color][2]  # 直接根据颜色取 label

        if label not in face_instances:
            face_instances[label] = []

        # 注意：此时 face 是原始 mesh 的顶点索引，我们稍后再映射到局部
        face_instances[label].append(face)
    else:
        print(f"Face color {face_color} not found in color2label mapping.")

for label, faces in face_instances.items():
    # 收集所有该 label 用到的顶点索引
    vertex_indices = set([vid for f in faces for vid in f])

    # 构建原始顶点到局部顶点的映射
    vertex_idx_map = {v: i for i, v in enumerate(vertex_indices)}

    # 构建局部顶点和三角形索引
    vertice_node = np.array([label_trimesh.vertices[v] for v in vertex_indices])
    face_node = np.array([[vertex_idx_map[v] for v in f] for f in faces])

    label_color = label_color_map[label]
    vertice_color_node = np.array([label_color] * vertice_node.shape[0])
    face_color_node = np.array([label_color] * face_node.shape[0])

    mesh_node = trimesh.Trimesh(vertices=vertice_node, faces=face_node, vertex_colors=vertice_color_node, face_colors=face_color_node)

    # 当前模型添加到场景中
    node = label_scene.add(pyrender.Mesh.from_trimesh(mesh_node))
    seg_node_map[node] = label_color




# %%
