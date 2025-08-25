import torch
import cv2
import numpy as np
from torchvision.transforms import ToTensor

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
        """
        Here we assume the batch_size = 1

        Args:
        batch: list of dict with keys: file_name, height, width, image_id, image
            Example:
            file_name: "./datasets/teeth3ds/teeth3ds_coco36/test/4MC4KRQV_upper_0.png"
            height, width: 1024 1024
            image: torch.Size([3, 1024, 1024])

        Returns: list of prediction with the following form
        {
            "masks": [B, N, H, W] binary masks
            "boxes": [B, N, 4] boxes in (x1, y1, x2, y2) format
            "box_scores": [B, N] confidence scores
            "classes": [B, N] class labels
        }
        """
        image = cv2.imread(batch[0]["file_name"])
        boxes, scores, classes = self.detector(image)
        # print(f"Detected boxes: {boxes.shape}, scores: {scores.shape}, classes: {classes.shape}")
        
        batched_points, batched_labels = boxes_to_points_labels(boxes)
        # print(f"Converted points: {batched_points.shape}, labels: {batched_labels.shape}")

        if isinstance(image, np.ndarray):
            batched_images = ToTensor()(image).unsqueeze(0)  # [1,3,H,W]
        else:
            batched_images = image.unsqueeze(0)
        batched_images = batched_images.to(self.device)
        # print(f"Image batch shape: {batched_images.shape}")

        results = self.segmenter(batched_images, batched_points, batched_labels)
        predicted_logits, predicted_iou = results
        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
        predicted_logits = torch.take_along_dim(
            predicted_logits, sorted_ids[..., None, None], dim=2
        )
        # print(f"Segmented masks: {predicted_logits.shape}, IOU: {predicted_iou.shape}")

        # threshold 0, add batch dimensions
        masks = torch.ge(predicted_logits[:, :, 0, :, :], 0)
        boxes = boxes.unsqueeze(0)
        scores = scores.unsqueeze(0)
        classes = classes.unsqueeze(0)

        return {
            "masks": masks,  # [B, N, H, W]
            "boxes": boxes,       # [B, N, 4]
            "box_scores": scores,     # [B, N]
            "classes": classes,   # [B, N]
        }