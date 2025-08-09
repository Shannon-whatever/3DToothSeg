import numpy as np
from plyfile import PlyData
from sklearn.neighbors import NearestNeighbors
from utils.other_utils import color2label
import torchmetrics as tm
import torch

def read_ply_face_center_and_labels(ply_path):
    plydata = PlyData.read(ply_path)
    face = plydata['face']
    vertex = plydata['vertex']
    vertices = np.vstack((vertex['x'], vertex['y'], vertex['z'])).T  # (V, 3)
    face_indices = face['vertex_indices']  # (F, 3)
    # calculate face center coordinate
    face_centers = np.array([vertices[idxs].mean(axis=0) for idxs in face_indices])  # (F, 3)
    # mapping color to label
    colors = np.stack([face['red'], face['green'], face['blue']], axis=1)
    labels = np.array([color2label[tuple(c)][2] for c in colors], dtype=np.int32)
    return face_centers, labels

def calculate_miou(pred_labels, gt_labels, n_class=17, ignore_index=-1,
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


    # valid_mask = (gt_labels != ignore_index)

    # pred_masked = pred_labels.clone()
    # gt_masked = gt_labels.clone()

    # pred_masked = pred_masked[valid_mask].reshape(bs, -1)
    # gt_masked = gt_masked[valid_mask].reshape(bs, -1)
    
    cal_miou = tm.JaccardIndex(task="multiclass", num_classes=n_class, ignore_index=ignore_index).to(device)
    miou = cal_miou(pred_labels, gt_labels)

    cal_iou = tm.JaccardIndex(task="multiclass", num_classes=n_class, average=None, ignore_index=ignore_index).to(device)
    per_class_iou = cal_iou(pred_labels, gt_labels) # (C, )

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

def calculate_miou_2d(pred_labels, gt_labels, n_class=17+1, ignore_index=-1):
    device = gt_labels.device
    miou_metric = tm.JaccardIndex(task="multiclass", num_classes=n_class, ignore_index=ignore_index)

    # 计算 mIoU
    miou = miou_metric(pred_labels, gt_labels)

    cal_iou = tm.JaccardIndex(task="multiclass", num_classes=n_class, average=None, ignore_index=ignore_index).to(device)
    per_class_iou = cal_iou(pred_labels, gt_labels) # (C, )

    return miou, per_class_iou



def cal_weighted_miou(gt_labels, pred_labels, n_class=17):
    pred_labels = torch.tensor(pred_labels)
    gt_labels = torch.tensor(gt_labels)


    cal_iou = tm.JaccardIndex(task="multiclass", num_classes=n_class, average=None)
    per_class_iou = cal_iou(pred_labels, gt_labels)

    iou_0 = per_class_iou[0]

    # 3. 计算各类别在gt中的像素数（权重）
    gt_labels_flat = gt_labels.view(-1)
    counts = torch.bincount(gt_labels_flat, minlength=n_class).float()
    weights = counts / counts.sum()

    # 4. 计算加权mIoU（忽略没出现的类别）
    mask = counts > 0  # 只对出现过的类别加权
    weighted_iou = per_class_iou[mask] * weights[mask]
    cal_weighted_miou = weighted_iou.sum()

    return cal_weighted_miou, iou_0

# def calculate_per_class_iou(gt_labels, pred_labels, n_class=17):
#     pred_labels = torch.as_tensor(pred_labels)
#     gt_labels = torch.as_tensor(gt_labels)

#     cal_iou = tm.JaccardIndex(task="multiclass", num_classes=n_class, average=None)
#     per_class_iou = cal_iou(pred_labels, gt_labels)
#     return per_class_iou


def compute_boundary_mask(face_centers, labels, threshold=4, k=8):
    """
    向量化计算边界点mask。
    - face_centers: (F, 3)
    - labels: (F,)
    返回 bool数组，True表示边界点
    """
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(face_centers)
    _, indices = nbrs.kneighbors(face_centers)  # (F, k+1)
    
    # 取出邻居标签，shape=(F, k+1)
    neighbor_labels = labels[indices]

    # 中心点标签扩展到邻居维度，shape=(F, 1)
    center_labels = labels[:, np.newaxis]

    # 判断邻居标签是否与中心点标签相等，shape=(F, k+1)
    equal_mask = (neighbor_labels == center_labels)

    # 排除自己（第0个邻居即自身）
    equal_mask = equal_mask[:, 1:]  # shape=(F, k)

    # 计算与中心点标签不相等的邻居数量
    diff_count = np.sum(~equal_mask, axis=1)  # shape=(F,)

    # 多数邻居标签不同则为边界点
    boundary_mask = diff_count > threshold

    return boundary_mask


# def calculate_merged_ious(gt_labels, pred_labels, eps=True):
#     merged_ious = {}
#     merge_pairs = [
#         (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15), (8, 16)
#     ]

#     for a, b in merge_pairs:
#         gt_merge = (gt_labels == a) | (gt_labels == b)
#         pred_merge = (pred_labels == a) | (pred_labels == b)
#         intersection = np.logical_and(gt_merge, pred_merge).sum()
#         union = np.logical_or(gt_merge, pred_merge).sum()

#         if eps:
#             iou = intersection / (union + 1e-6)
#             merged_ious[f"T{a}/T{b}"] = iou
#         else:
#             if union != 0:
#                 iou = intersection / union
#                 merged_ious[f"T{a}/T{b}"] = iou

#     return merged_ious