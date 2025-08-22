import torch
import os
import torchmetrics as tm
import numpy as np
from PIL import Image
from utils.color_utils import color2label

def masks_to_label_map_batch(mask, classes, valid_mask, ignore_index=0):
    """
    Convert batched masks to label map.
    
    mask: [B, N, H, W] predicted mask probabilities (0~1)
    classes: [B, N] predicted class IDs
    valid_mask: [B, N] boolean mask for valid boxes/masks
    
    Returns:
        label_map: [B, H, W] predicted label for each pixel
    """
    B, N, H, W = mask.shape
    device = mask.device
    label_map = torch.full((B, H, W), ignore_index, dtype=torch.long, device = device)

    for b in range(B):
        for n in range(N):
            if valid_mask[b, n]:
                mask_n = mask[b, n] > 0.5  # threshold
                label_map[b][mask_n] = classes[b, n] + 1
    
    return label_map

def color_mask_to_label_maps_batch(batch, ignore_index=0, device="cpu"):
    """
    Convert a batch of ground truth RGB mask images to label maps.

    Args:
        batch: list of dicts, each dict has key 'file_name' (path to input image)
        ignore_index: class index for background or unknown pixels
        device: "cpu" or "cuda"

    Returns:
        torch.Tensor of shape [B, H, W], dtype=torch.long
    """
    label_maps = []

    for item in batch:
        file_name = os.path.basename(item['file_name'])
        mask_path = os.path.join("./datasets/teeth3ds/teeth3ds_coco36", "test_mask", file_name)

        # Load RGB mask
        mask = np.array(Image.open(mask_path).convert("RGB"))
        H, W, _ = mask.shape

        # Initialize label map
        label_map = np.full((H, W), ignore_index, dtype=np.int64)

        # Assign class ids based on color2label
        for color, (_, _, class_id) in color2label.items():
            if class_id == 0:
                continue
            matches = np.all(mask == np.array(color, dtype=np.uint8), axis=-1)
            label_map[matches] = class_id 

        label_maps.append(label_map)

    # Ensure same H, W across the batch
    H, W = label_maps[0].shape
    for lm in label_maps:
        assert lm.shape == (H, W), "All GT masks must have the same size for batching"

    # Stack and move to device
    label_maps = np.stack(label_maps, axis=0)   # [B, H, W]
    return torch.from_numpy(label_maps).to(device=device, dtype=torch.long)

def calculate_miou_2d(pred_labels, gt_labels, n_class=16+1, ignore_index=0):
    device = gt_labels.device
    # print(f"shape of pred_labels: {pred_labels.shape}, gt_labels: {gt_labels.shape}")
    # print(torch.unique(gt_labels))
    # print(torch.unique(pred_labels))

    miou_metric = tm.JaccardIndex(task="multiclass", num_classes=n_class, ignore_index=ignore_index).to(device)

    miou = miou_metric(pred_labels, gt_labels)

    cal_iou = tm.JaccardIndex(task="multiclass", num_classes=n_class, average=None, ignore_index=ignore_index).to(device)
    per_class_iou = cal_iou(pred_labels, gt_labels) # (C, )

    return miou, per_class_iou