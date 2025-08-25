import torch
import numpy as np
from torchvision.transforms import ToTensor
from .detector import FasterRCNNDetector
from .segmenter import EfficientSAMSegmenter

class DetectionSegmentationPipeline:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.detector = FasterRCNNDetector(config_path="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", 
                 weights_path="model_final.pth", 
                 score_thresh=0.5)
        self.segmenter = EfficientSAMSegmenter()
    
    def boxes_to_points_labels(self, boxes):
        """
        Convert [N,4] boxes to batched_points and batched_point_labels tensors for EfficientSAM
        """
        num_boxes = boxes.shape[0]

        points_list = []
        labels_list = []

        for i in range(num_boxes):
            x1, y1, x2, y2 = boxes[i]
            points_list.append([[x1, y1], [x2, y2]])    # 2 points per box
            labels_list.append([2, 3])                  # 2=top-left, 3=bottom-right

        batched_points = torch.tensor(points_list).unsqueeze(0)  # [B=1, num_queries=num_boxes, num_pts=2, 2]
        batched_point_labels = torch.tensor(labels_list).unsqueeze(0)  # [B=1, num_queries=num_boxes, num_pts=2]

        return batched_points, batched_point_labels

    def run(self, image):
        # Step 1: detect objects
        boxes, scores, classes = self.detector.predict(image)

        # Step 2: convert boxes to points/labels for EfficientSAM
        batched_points, batched_point_labels = self.boxes_to_points_labels(boxes)

        # Step 3: convert NumPy image to tensor if needed
        if isinstance(image, np.ndarray):
            batched_images = ToTensor()(image).unsqueeze(0)  # [1,3,H,W]
        else:
            batched_images = image.unsqueeze(0)

        # Step 4: move everything to the device
        batched_images = batched_images.to(self.device)
        batched_points = batched_points.to(self.device)
        batched_point_labels = batched_point_labels.to(self.device)

        # Step 5: feed boxes into EfficientSAM
        results = self.segmenter.segment(
            batched_images,
            batched_points,
            batched_point_labels
        )

        predicted_logits, predicted_iou = results
        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
        predicted_logits = torch.take_along_dim(
            predicted_logits, sorted_ids[..., None, None], dim=2
        )

        return {
            "masks": torch.ge(predicted_logits[:, :, 0, :, :], 0).cpu().detach().numpy(),  # [B, N, H, W]
            "boxes": boxes,       # [B, N, 4]
            "box_scores": scores,     # [B, N]
            "classes": classes,   # [B, N]
        }