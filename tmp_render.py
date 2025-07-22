import math
import os
import os.path
os.environ["PYOPENGL_PLATFORM"] = "egl"

import matplotlib.pyplot as plt
import numpy as np
import pyrender
import trimesh
from utils.other_utils import color2label


def m3dLookAt(eye, target, up):
    def normalize(v):
        return v / np.sqrt(np.sum(v ** 2))

    mz = normalize(eye - target)
    mx = normalize(np.cross(up, mz))
    my = normalize(np.cross(mz, mx))

    return np.array([
        [mx[0], my[0], mz[0], eye[0]],
        [mx[1], my[1], mz[1], eye[1]],
        [mx[2], my[2], mz[2], eye[2]],
        [0, 0, 0, 1]
    ])


def render(model_path, save_path, rend_size=(256, 256), rend_step=(9, 12)):
    base_name = os.path.basename(model_path)[:-4]

    os.makedirs(os.path.join(save_path, 'render'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'mask'), exist_ok=True)

    label_trimesh = trimesh.load(model_path)
    # fuze_trimesh = trimesh.load(model_path)
    vertices = np.asarray(label_trimesh.vertices)
    minCoord = np.min(vertices, axis=0)
    maxCoord = np.max(vertices, axis=0)
    meanCoord = np.mean(vertices, axis=0)
    x_len, y_len, z_len = maxCoord - minCoord
    # 计算球体半径
    radius = np.sqrt(np.sum((maxCoord - meanCoord) ** 2)) * 1.2
    theta = math.asin((z_len / 2) / radius)

    # 记录所有的相机位姿
    camera_pos_list = []

    while theta <= math.pi / 2:
        z_offset = math.sin(theta) * radius
        sub_radius = math.cos(theta) * radius
        beta = 0
        while beta <= math.pi * 2:
            x_offset = sub_radius * math.cos(beta)
            y_offset = sub_radius * math.sin(beta)
            camera_pos = meanCoord + np.asarray([x_offset, y_offset, z_offset], dtype=float)

            # 计算一个上方向
            camera_pos_list.append((
                # 摄像机朝向
                m3dLookAt(camera_pos, meanCoord, np.asarray([0, 0, 1], dtype=float)),
                theta,
                beta
            ))
            beta += math.pi / rend_step[0]
        theta += math.pi / rend_step[1]

    # 创建模型
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

    face_instances = {}
    if 'face' in label_trimesh.metadata['_ply_raw']:
        face_meta = label_trimesh.metadata['_ply_raw']['face']
        if 'red' in face_meta['data'] and 'green' in face_meta['data'] and 'blue' in face_meta['data']:
            face_colors = np.stack([
                face_meta['data']['red'],
                face_meta['data']['green'],
                face_meta['data']['blue']
            ], axis=-1).squeeze(1)  # shape: (N_faces, 3)

    for i, face in enumerate(label_trimesh.faces):
        face_color = tuple(face_colors[i])  # 取RGB，忽略alpha

        if face_color in color2label:
            label = color2label[face_color][2]  # 直接根据颜色取 label

            if label not in face_instances:
                face_instances[label] = []

            # 注意：此时 face 是原始 mesh 的顶点索引，我们稍后再映射到局部
            face_instances[label].append(face)
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



    # for i, face in enumerate(label_trimesh.faces):
    #     for label, vertices in vertex_instances.items():
    #         if face[0] in vertices and face[1] in vertices and face[2] in vertices:
    #             if not label in face_instances:
    #                 face_instances[label] = []
    #             face_instances[label].append([vertices[face[0]], vertices[face[1]], vertices[face[2]]])
    # for label, vertices in vertex_instances.items():
    #     label_color = label_color_map[label]
    #     label_color = [label_color[0], label_color[1], label_color[2]]
    #     vertice_node = np.array([label_trimesh.vertices[i] for i, _ in vertices.items()], dtype=float)
    #     vertice_color_node = np.array([label_color] * vertice_node.shape[0])
    #     face_node = np.array(face_instances[label])
    #     face_color_node = np.array([label_color] * face_node.shape[0])
    #     mesh_node = trimesh.Trimesh(vertices=vertice_node, faces=face_node, vertex_colors=vertice_color_node, face_colors=face_color_node)

    #     # 当前模型添加到场景中
    #     node = label_scene.add(pyrender.Mesh.from_trimesh(mesh_node))
    #     seg_node_map[node] = label_color

    # 添加光源
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light)
    label_scene.add(light)

    # 创建相机
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2, aspectRatio=1.0)

    # 渲染参数
    r = pyrender.OffscreenRenderer(viewport_width=rend_size[1],
                                   viewport_height=rend_size[0],
                                   point_size=1.0)

    angle_info = ""
    for i, camera_pos in enumerate(camera_pos_list):
        angle_info += f'theta {camera_pos[1]} beta {camera_pos[2]}\n'

        # 渲染图片
        camera_node = scene.add(camera, pose=camera_pos[0])
        color, depth = r.render(scene)
        scene.remove_node(camera_node)
        plt.imsave(os.path.join(save_path, 'render', f'{base_name}_{i}.png'), color)

        # 渲染分割标签
        camera_node = label_scene.add(camera, pose=camera_pos[0])
        color, depth = r.render(label_scene, flags=pyrender.RenderFlags.SEG, seg_node_map=seg_node_map)
        label_scene.remove_node(camera_node)
        plt.imsave(os.path.join(save_path, 'mask', f'{base_name}_{i}.png'), color)


    with open(os.path.join(os.path.join(save_path, f'{base_name}_render_view.txt')), 'w',
              encoding='ascii') as f:
        f.write(angle_info)

    r.delete()


if __name__ == '__main__':

    ply_cell_color_path = 'tmp/YBSESUN6_upper_gt.ply'
    save_path = "tmp"
    render(ply_cell_color_path, save_path, rend_size=(1024, 1024), rend_step=(6, 9))


