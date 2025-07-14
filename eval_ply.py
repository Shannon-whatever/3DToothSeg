import numpy as np
from plyfile import PlyData
import argparse
import sys
import os
import torch
from scipy import ndimage
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import color2label, cal_metric

def read_ply_face_labels_and_xyz(ply_path):
    """
    读取PLY文件中每个面的颜色标签和顶点坐标，并映射为类别索引
    返回：labels, face_xyzs
    """
    plydata = PlyData.read(ply_path)
    face = plydata['face']
    vertex = plydata['vertex']
    # 获取所有顶点坐标
    vertices = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
    # 获取面顶点索引
    face_indices = face['vertex_indices']
    # 获取面颜色
    if 'red' in face and 'green' in face and 'blue' in face:
        colors = np.stack([face['red'], face['green'], face['blue']], axis=1)
    else:
        raise ValueError("PLY文件的face没有颜色属性")
    labels = []
    face_xyzs = []
    for idxs, c in zip(face_indices, colors):
        idxs = list(idxs)
        # 按xyz排序顶点
        pts = vertices[idxs]
        pts_sorted = pts[np.lexsort((pts[:,2], pts[:,1], pts[:,0]))]
        face_xyzs.append(pts_sorted.flatten())
        c_tuple = tuple(c)
        if c_tuple in color2label:
            labels.append(color2label[c_tuple][2])
        else:
            labels.append(0)
    return np.array(labels, dtype=np.int32), np.array(face_xyzs)

def calculate_miou(gt_labels, pred_labels, n_class):
    ious = []
    weights = []
    for c in range(1, n_class):  # 排除背景0
        gt_mask = (gt_labels == c)
        pred_mask = (pred_labels == c)
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        iou = intersection / (union + 1e-6) if union > 0 else 1.0
        ious.append(iou)
        weights.append(gt_mask.sum())
    weights = np.array(weights, dtype=np.float32)
    if weights.sum() > 0:
        weighted_miou = np.sum(np.array(ious) * weights) / weights.sum()
    else:
        weighted_miou = 0.0
    return weighted_miou, ious, weights

def calculate_boundary_iou(gt_mask, pred_mask, boundary_width=3):
    gt_boundary = np.zeros_like(gt_mask, dtype=bool)
    pred_boundary = np.zeros_like(pred_mask, dtype=bool)
    for i in range(1, len(gt_mask)):
        if gt_mask[i] != gt_mask[i-1]:
            start = max(0, i - boundary_width)
            end = min(len(gt_mask), i + boundary_width)
            gt_boundary[start:end] = True
    for i in range(1, len(pred_mask)):
        if pred_mask[i] != pred_mask[i-1]:
            start = max(0, i - boundary_width)
            end = min(len(pred_mask), i + boundary_width)
            pred_boundary[start:end] = True
    intersection = np.logical_and(gt_boundary, pred_boundary)
    union = np.logical_or(gt_boundary, pred_boundary)
    return np.sum(intersection) / (np.sum(union) + 1e-6)

def calculate_biou(gt_labels, pred_labels, n_class, boundary_width=3):
    bious = []
    for c in range(1, n_class):  # 排除背景0
        gt_mask = (gt_labels == c)
        pred_mask = (pred_labels == c)
        biou = calculate_boundary_iou(gt_mask, pred_mask, boundary_width)
        bious.append(biou)
    return np.mean(bious) if bious else 0.0, bious

def main(pred_ply, gt_ply):
    print(f"预测文件: {pred_ply}")
    print(f"GT文件: {gt_ply}")
    pred_labels, pred_xyzs = read_ply_face_labels_and_xyz(pred_ply)
    gt_labels, gt_xyzs = read_ply_face_labels_and_xyz(gt_ply)
    if len(pred_labels) != len(gt_labels):
        print(f"错误：预测和GT的面数量不一致！pred={len(pred_labels)}, gt={len(gt_labels)}")
        return
    print(f"面数: {len(pred_labels)}")
    # 对面按xyz排序
    pred_order = np.lexsort(pred_xyzs.T)
    gt_order = np.lexsort(gt_xyzs.T)
    pred_labels_sorted = pred_labels[pred_order]
    gt_labels_sorted = gt_labels[gt_order]
    # 转为torch
    n_class = max(int(np.max(gt_labels_sorted)), int(np.max(pred_labels_sorted))) + 1
    gt_labels_torch = torch.from_numpy(gt_labels_sorted).long()
    pred_onehot = torch.from_numpy(np.eye(n_class)[pred_labels_sorted]).float()
    # 计算整体指标
    print("整体评估指标：")
    # 计算mIoU
    weighted_miou, class_ious, weights = calculate_miou(gt_labels_sorted, pred_labels_sorted, n_class)
    print(f"  weighted_mIOU: {weighted_miou:.4f}")
    # 计算bIoU
    biou, class_bious = calculate_biou(gt_labels_sorted, pred_labels_sorted, n_class, boundary_width=3)
    print(f"  bIOU    : {biou:.4f}")
    
    # 计算background IOU
    gt_background = (gt_labels_sorted == 0).astype(np.int32)
    pred_background = (pred_labels_sorted == 0).astype(np.int32)
    intersection = np.logical_and(gt_background, pred_background).sum()
    union = np.logical_or(gt_background, pred_background).sum()
    background_iou = intersection / (union + 1e-6) if union > 0 else 1.0
    print(f"  background IOU: {background_iou:.4f}")

    # 牙齿对合并映射
    merge_pairs = [
        (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15)
    ]
    print("\nT对称合并类别IOU：")
    
    # 准备保存结果
    result_lines = []
    result_lines.append(f"预测文件: {os.path.basename(pred_ply)}")
    result_lines.append(f"真实标签: {os.path.basename(gt_ply)}")
    result_lines.append("评估结果:")
    result_lines.append(f"预测文件: {pred_ply}")
    result_lines.append(f"GT文件: {gt_ply}")
    result_lines.append(f"面数: {len(pred_labels)}")
    result_lines.append("整体评估指标：")
    result_lines.append(f"  weighted_mIOU: {weighted_miou:.4f}")
    result_lines.append(f"  bIOU    : {biou:.4f}")
    result_lines.append(f"  background IOU: {background_iou:.4f}")
    result_lines.append("")
    result_lines.append("T对称合并类别IOU：")
    
    for a, b in merge_pairs:
        gt_merge = ((gt_labels_sorted == a) | (gt_labels_sorted == b)).astype(np.int32)
        pred_merge = ((pred_labels_sorted == a) | (pred_labels_sorted == b)).astype(np.int32)
        gt_merge_torch = torch.from_numpy(gt_merge).long()
        pred_merge_onehot = torch.from_numpy(np.eye(2)[pred_merge]).float()
        _, _, _, _, iou, _ = cal_metric(
            gt_labels=gt_merge_torch,
            pred_labels=pred_merge_onehot,
            target_class=1
        )
        print(f"T{a}/T{b} IOU: {iou:.4f}")
        result_lines.append(f"T{a}/T{b} IOU: {iou:.4f}")
    
    # 保存结果到eval_results文件夹
    os.makedirs("eval_results", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pred_ply))[0]
    if base_name.endswith('_mask'):
        base_name = base_name[:-5]  # 移除_mask后缀
    result_file = os.path.join("eval_results", f"{base_name}_eval_result.txt")
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result_lines))
    
    print(f"\n结果已保存到: {result_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估预测PLY与GT PLY的面分割指标")
    parser.add_argument('--pred', type=str, default='predict.ply', help='预测PLY文件路径')
    parser.add_argument('--gt', type=str, default='YBSESUN6_upper_mask.ply', help='GT PLY文件路径')
    args = parser.parse_args()
    main(args.pred, args.gt) 