from tracemalloc import start
import torch

from .detector import FasterRCNNDetector
from .segmenter import EfficientSAMSegmenter
from utils import boxes_to_points_labels


class DetectionSegmentationPipeline:
    def __init__(self, cfg):
        self.cfg = cfg.clone() 
        self.device = cfg.MODEL.DEVICE
        self.detector = FasterRCNNDetector(cfg)
        self.segmenter = EfficientSAMSegmenter(cfg)

    def __call__(self, batch):
        # GPU timers
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        timings = {}

        # Step 1: Detection
        start.record()
        padded_boxes, padded_scores, padded_classes, valid_mask = self.detector(batch)
        # print(f"Detected boxes: {padded_boxes.shape}, scores: {padded_scores.shape}, classes: {padded_classes.shape}, valid mask: {valid_mask.shape}")
        end.record()
        torch.cuda.synchronize()
        timings['detection'] = start.elapsed_time(end)  # in milliseconds

        # Step 2: boxes → points
        start.record()
        batched_points, batched_labels = boxes_to_points_labels(padded_boxes)
        # print(f"Converted points: {batched_points.shape}, labels: {batched_labels.shape}")
        end.record()
        torch.cuda.synchronize()
        timings['boxes_to_points'] = start.elapsed_time(end)
        
        # Step 3: get batched_images [B,3,H,W]
        start.record()
        batched_images = torch.stack([x["image"] for x in batch]) 
        if batched_images.dtype != torch.float32:
            batched_images = batched_images.float()
        # print(f"Image batch shape: {batched_images.shape}")

        batched_images = torch.stack([x["image"] for x in batch]).pin_memory().to(self.device, non_blocking=True)
        batched_points = batched_points.to(self.device, non_blocking=True)
        batched_labels = batched_labels.to(self.device, non_blocking=True)

        end.record()
        torch.cuda.synchronize()
        timings['prepare_inputs'] = start.elapsed_time(end)

        # Step 4: segment
        start.record()
        results = self.segmenter(
            batched_images,
            batched_points,
            batched_labels
        )
        end.record()
        torch.cuda.synchronize()
        timings['segmentation'] = start.elapsed_time(end)

        # Step 5: choose best masks
        start.record()
        masks, iou_predictions = results  # shapes [B, N, 3, H, W] and [B, N, 3]
        best_idx = torch.argmax(iou_predictions, dim=-1)  # [B, N]
        B, N, _, _, _ = masks.shape
        best_masks = masks[torch.arange(B)[:, None], torch.arange(N)[None, :], best_idx]  # [B, N, H, W]
        best_ious = torch.max(iou_predictions, dim=-1).values  # [B, N]
        # print(f"Segmented masks: {best_masks.shape}, IOU predictions: {best_ious.shape}")
        end.record()
        torch.cuda.synchronize()
        timings['postprocess_masks'] = start.elapsed_time(end)

        # print(timings)

        return {
            "masks": best_masks,         # [B, N, H, W]
            "boxes": padded_boxes,       # [B, N, 4]
            "box_scores": padded_scores,     # [B, N]
            "iou_predictions": best_ious, #[B, N]
            "classes": padded_classes,   # [B, N]
            "valid_mask": valid_mask     # [B, N]
        }