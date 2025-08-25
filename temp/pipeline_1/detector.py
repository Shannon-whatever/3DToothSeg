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
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, image):
        """
        Arguments: image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
        Boxes.tensor: N x 4 tensor of format (x1, x2, y1, y2)
        scores: Tensor of N confidence scores
        pred_classes: Tensor of N labels in range [0, num_categories)
        """
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            if self.input_format == "RGB":
                image = image[:, :, ::-1]
            height, width = image.shape[:2]
            image = self.aug.get_transform(image).apply_image(image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image.to(self.cfg.MODEL.DEVICE)

            inputs = {"image": image, "height": height, "width": width}

            outputs = self.model([inputs])[0]
            instances = outputs["instances"]
            boxes = instances.pred_boxes.tensor
            scores = instances.scores
            classes = instances.pred_classes
            return boxes, scores, classes