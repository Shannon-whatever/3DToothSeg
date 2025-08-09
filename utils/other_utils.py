import torch
import numpy as np
from plyfile import PlyData
import os
from PIL import Image

from utils.color_utils import color2label


def cal_metric(gt_labels, pred_labels, target_class=None):
    pred_softmax = torch.nn.functional.softmax(pred_labels, dim=1)
    _, pred_classes = torch.max(pred_softmax, dim=1)

    if target_class is not None:
        gt_labels = (gt_labels == target_class)
        pred_classes = (pred_classes == target_class)

    correct_mask = pred_classes == gt_labels
    wrong_mask = pred_classes != gt_labels
    positive_mask = gt_labels != 0
    negative_mask = gt_labels == 0

    TP = torch.sum(positive_mask * correct_mask).item()
    TN = torch.sum(negative_mask * correct_mask).item()
    FP = torch.sum(positive_mask * wrong_mask).item()
    FN = torch.sum(negative_mask * wrong_mask).item()

    eps = 1e-6

    ACC = (TP + TN) / (FP + TP + FN + TN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    F1 = 2 * (precision * recall) / (precision + recall + eps)
    IOU = TP / (FP + TP + FN + eps)

    return ACC, precision, recall, F1, IOU, pred_classes


def output_pred_ply(pred_mask, cell_coords, path, point_coords=None, face_info=None, vertex_colors=None):
    if point_coords is not None and face_info is not None:
        vertex_info = ""
        cell_info = ""
        for idx, pc in enumerate(point_coords):
            if vertex_colors is None:
                vertex_info += f'{pc[0]} {pc[1]} {pc[2]} {125} {125} {125} {255}\n'
            else:
                vertex_info += f'{pc[0]} {pc[1]} {pc[2]} {vertex_colors[idx][0]} {vertex_colors[idx][1]} {vertex_colors[idx][2]} {255}\n'

        valid_face_num = 0
        for color, fi in zip(pred_mask, face_info):
            if fi[0] == 0 and fi[1] == 0 and fi[2] == 0:
                continue
            cell_info += f'3 {int(fi[0])} {int(fi[1])} {int(fi[2])} {color[0]} {color[1]} {color[2]} {255}\n'
            valid_face_num += 1
        header = (f"ply\n"
                  f"format ascii 1.0\n"
                  f"comment VCGLIB generated\n"
                  f"element vertex {point_coords.shape[0]}\n"
                  f"property double x\n"
                  f"property double y\n"
                  f"property double z\n"
                  f"property uchar red\n"
                  f"property uchar green\n"
                  f"property uchar blue\n"
                  f"property uchar alpha\n"
                  f"element face {valid_face_num}\n"
                  f"property list uchar int vertex_indices\n"
                  f"property uchar red\n"
                  f"property uchar green\n"
                  f"property uchar blue\n"
                  f"property uchar alpha\n"
                  f"end_header\n")
    else:
        header = (f"ply\n"
                  f"format ascii 1.0\n"
                  f"comment VCGLIB generated\n"
                  f"element vertex {cell_coords.shape[0]}\n"
                  f"property double x\n"
                  f"property double y\n"
                  f"property double z\n"
                  f"property uchar red\n"
                  f"property uchar green\n"
                  f"property uchar blue\n"
                  f"property uchar alpha\n"
                  f"element face {0}\n"
                  f"property list uchar int vertex_indices\n"
                  f"property uchar red\n"
                  f"property uchar green\n"
                  f"property uchar blue\n"
                  f"property uchar alpha\n"
                  f"end_header\n")

        vertex_info = ""
        cell_info = ""
        for color, coord in zip(pred_mask, cell_coords):
            vertex_info += f'{coord[0]} {coord[1]} {coord[2]} {color[0]} {color[1]} {color[2]} {255}\n'

    with open(path, 'w', encoding='ascii') as f:
        f.write(header)
        f.write(vertex_info)
        f.write(cell_info)

    return


def output_pred_images(pred_rgb, gt_rgb, save_dir, file_name):
    """
    pred_rgb: (view, h, w, 3) uint8 tensor
    """

    view = pred_rgb.shape[0]

    for v in range(view):
        pred_img = pred_rgb[v]  # (h, w, 3)
        pred_img = Image.fromarray(pred_img)
        pred_img_save_dir = os.path.join(save_dir, 'pred_mask')
        os.makedirs(pred_img_save_dir, exist_ok=True)
        pred_img.save(os.path.join(pred_img_save_dir, f"{file_name}_{v}.png"))

        gt_img = gt_rgb[v]  # (h, w, 3)
        gt_img = Image.fromarray(gt_img)
        gt_img_save_dir = os.path.join(save_dir, 'gt_mask')
        os.makedirs(gt_img_save_dir, exist_ok=True)
        gt_img.save(os.path.join(gt_img_save_dir, f"{file_name}_{v}.png"))


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

    return np.array(labels)


from scipy.spatial import KDTree

def get_boundary_mask(vertices, vertex_labels, k=8):
    tree = KDTree(vertices)
    dists, idxs = tree.query(vertices, k=k+1)  # k+1 因为包含自身

    boundary_mask = np.zeros(len(vertices), dtype=np.uint8)  # 0/1 mask

    for i in range(len(vertices)):
        neighbors = idxs[i, 1:]  # 去除自身
        neighbor_labels = vertex_labels[neighbors]
        center_label = vertex_labels[i]
        diff_count = np.sum(neighbor_labels != center_label)

        if diff_count >= (k // 2 + 1):  # 超过一半邻居类别不同
            boundary_mask[i] = 1

    return boundary_mask


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


def rgb_mask_to_label(mask_tensor, num_classes=17):
    """
    将 RGB mask 转换为 one-hot 编码 label tensor。
    
    参数:
        mask_tensor: torch.Tensor, shape=(3, H, W)
        color2label: dict, key=(R,G,B) tuple, value=label id
        num_classes: int, 类别总数

    返回:
        label_map: torch.Tensor, shape=(H, W), dtype=torch.long
    """
    H, W = mask_tensor.shape[1], mask_tensor.shape[2]
    label_map = torch.full((H, W), fill_value=num_classes, dtype=torch.long) # 17 is background

    # 转换成 (H, W, 3)
    mask_np = mask_tensor.permute(1, 2, 0).cpu().numpy()

    # 遍历 color2label 映射
    for color, label_id in color2label.items():
        match = (mask_np == color).all(axis=-1)  # shape: (H, W), bool
        # print(label_id, np.any(match), np.count_nonzero(match))
        label_map[match] = label_id[-1]

    # # 检查是否有未映射像素
    # if (label_map == -1).any():
    #     print("Warning: some pixels do not match any color in color2label")

    # # 转 one-hot
    # one_hot = torch.nn.functional.one_hot(label_map.clamp(min=0), num_classes=num_classes+1)  # (H, W, C)
    # one_hot = one_hot.permute(2, 0, 1)  # (C, H, W)
    return label_map



def save_metrics_to_txt(filepath, num_classes, miou, per_class_miou, merge_iou, 
                        merge_pairs=[(1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15), (8, 16)]):
    """
    Save evaluation metrics to a plain text (.txt) file.

    Args:
        filepath: str, path to save the txt file
        miou: float, overall mean IoU
        per_class_miou: Tensor (C,)
        merge_iou: Tensor (num_pairs,)
        class_names: Optional[List[str]], names for each class
        merge_names: Optional[List[str]], names for each merge pair
    """
    class_names = [f"Class {i}" for i in range(num_classes)]
    merge_names = [f"T{a}/T{b}" for a, b in merge_pairs]
    with open(filepath, "w") as f:
        f.write("==== Segmentation Evaluation Metrics ====\n\n")
        f.write(f"Overall mIoU: {miou:.4f}\n\n")

        f.write("Per-Class mIoU:\n")
        for i, iou in enumerate(per_class_miou):
            name = class_names[i] if class_names else f"Class {i}"
            f.write(f"  {name:<10s}: {iou:.4f}\n")
        f.write("\n")

        if merge_iou is not None:
            f.write("Merged-Class IoU:\n")
            for i, iou in enumerate(merge_iou):
                name = merge_names[i] if merge_names else f"Pair {i}"
                f.write(f"  {name:<10s}: {iou:.4f}\n")

