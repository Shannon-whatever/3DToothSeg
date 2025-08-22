import os
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

class FasterRCNNDetector:
    def __init__(self, config_path="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", 
                 weights_path="model_final.pth", score_thresh=0.5):
        # LOAD CONFIG
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config_path))
        # CONFIG SETTING
        cfg.DATASETS.TEST = ("teeth3ds_coco36_test",)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16
        cfg.OUTPUT_DIR = "./output_1"
        cfg.INPUT.MAX_SIZE_TEST = 1024
        cfg.INPUT.MIN_SIZE_TEST = 1024
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        if weights_path is not None:
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, weights_path)
        else:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        # BUILD PREDICTOR
        self.predictor = DefaultPredictor(cfg)

    def predict(self, image):
        outputs = self.predictor(image)
        # Arguments: original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        # Returns: list[dict]
        # instances object with the following fields
        # pred_boxes: Boxes object containing N boxes (see structure.boxes.py)
        # Boxes.tensor: N x 4 tensor of format (x1, x2, y1, y2)
        # scores: Tensor of N confidence scores
        # pred_classes: Tensor of N labels in range [0, num_categories)
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        return boxes, scores, classes

