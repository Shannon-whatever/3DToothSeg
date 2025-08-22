import os
import torch

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances

def setup_cfg(mode):
    # REGISTER DATASETS # TODO
    register_coco_instances("teeth3ds_coco36_train", {}, "./datasets/teeth3ds/teeth3ds_coco36/train_annotation.json", "./datasets/teeth3ds/teeth3ds_coco36/train")
    register_coco_instances("teeth3ds_coco36_test", {}, "./datasets/teeth3ds/teeth3ds_coco36/test_annotation.json", "./datasets/teeth3ds/teeth3ds_coco36/test")

    # CONFIG
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16
    cfg.OUTPUT_DIR = "./output_1"
    
    if mode == "train":
        # DATASETS
        cfg.DATASETS.TRAIN = ("teeth3ds_coco36_train",)
        # DATALOADER
        cfg.DATALOADER.NUM_WORKERS = 4
        # MODELS
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml')
        # SOLVER
        cfg.SOLVER.IMS_PER_BATCH = 36
        cfg.SOLVER.BASE_LR = 0.045
        cfg.SOLVER.WARMUP_ITERS = 1000
        cfg.SOLVER.MAX_ITER = 28800
        cfg.SOLVER.STEPS = (23040, 25920)
        cfg.SOLVER.CHECKPOINT_PERIOD = 14400

    elif mode == "infer" or mode == "eval":
        cfg.DATASETS.TEST = ("teeth3ds_coco36_test",)
        cfg.INPUT.MAX_SIZE_TEST = 1024
        cfg.INPUT.MIN_SIZE_TEST = 1024
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg