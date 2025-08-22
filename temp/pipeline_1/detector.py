import torch

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog

"""
Modified from DefaultPredictor from https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/defaults.py
Accepts dataloader, supports batching
"""
class FasterRCNNDetector:
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        self.cfg = cfg.clone() 
        self.device = cfg.MODEL.DEVICE 
        self.model = build_model(self.cfg)

        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], 
            cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, batch):
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch)
            # Args: batched_inputs: List[Dict[str, torch.Tensor]]
            # For now, each item in the list is a dict that contains:
            # image: Tensor, image in (C, H, W) format.
            # instances (optional): groundtruth :class:`Instances`

            # Returns: list[dict]
            # instances object with the following fields
            # pred_boxes: Boxes object containing N boxes (see structure.boxes.py)
            # Boxes.tensor: N x 4 tensor of format (x1, x2, y1, y2)
            # scores: Tensor of N confidence scores
            # pred_classes: Tensor of N labels in range [0, num_categories)

            all_boxes, all_scores, all_classes = [], [], []

            for output in outputs:
                instances = output["instances"]
                boxes = instances.pred_boxes.tensor     # [Ni, 4]
                scores = instances.scores               # [Ni]
                classes = instances.pred_classes        # [Ni]
                # print(f"boxes shape: {boxes.shape}, scores shape: {scores.shape}, classes shape: {classes.shape}")

                all_boxes.append(boxes) 
                all_scores.append(scores) 
                all_classes.append(classes) 

            B = len(outputs)
            max_num_boxes = max(boxes.shape[0] for boxes in all_boxes)
            # PADDING
            padded_boxes = torch.zeros(B, max_num_boxes, 4)
            padded_scores = torch.zeros(B, max_num_boxes)
            padded_classes = torch.zeros(B, max_num_boxes, dtype=torch.long)
            valid_mask = torch.zeros(B, max_num_boxes, dtype=torch.bool)

            for i in range(B):
                n = all_boxes[i].shape[0]
                padded_boxes[i, :n] = all_boxes[i]
                padded_scores[i, :n] = all_scores[i]
                padded_classes[i, :n] = all_classes[i]
                valid_mask[i, :n] = True

            return padded_boxes, padded_scores, padded_classes, valid_mask
            # padded_boxes: [B, max_num_boxes, 4]
            # padded_scores: [B, max_num_boxes]
            # padded_classes: [B, max_num_boxes]
            # valid_mask: [B, max_num_boxes]