import argparse
import random
import cv2
import os
import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid")

# DATASETS
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
# CONFIG
from detectron2.config import get_cfg
from detectron2 import model_zoo
# MODELS
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
# VISUALIZATION
from detectron2.utils.visualizer import Visualizer

def setup_cfg(mode, dataset):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16
    cfg.OUTPUT_DIR = "./output_1"   # single place for all outputs
    
    if mode == "train":
        # DATASETS
        cfg.DATASETS.TRAIN = ("teeth3ds_coco36_train",) if dataset == "teeth3ds_coco36" else ("teeth3ds_coco_train",)
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
        cfg.DATASETS.TEST = ("teeth3ds_coco36_test",) if dataset == "teeth3ds_coco36" else ("teeth3ds_coco_test",)
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def train(args):
    cfg = setup_cfg("train", args.dataset)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

def infer(args):
    cfg = setup_cfg("infer", args.dataset)
    predictor = DefaultPredictor(cfg)

    teeth3ds_coco36_test_metadata = MetadataCatalog.get("teeth3ds_coco36_test")
    test_dataset_dicts = DatasetCatalog.get("teeth3ds_coco36_test")

    for idx, d in enumerate(random.sample(test_dataset_dicts, 3)):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                    metadata=teeth3ds_coco36_test_metadata, 
                    scale=1.0, 
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        result = v.get_image()[:, :, ::-1]

        save_dir = "/home/zychen/Documents/Project_shno/3DToothSeg/temp/output_vis"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"pred_{idx}.jpg")
        ok = cv2.imwrite(save_path, result)
        print(f"Saved {save_path}, success={ok}")

def eval(args):
    cfg = setup_cfg("eval", args.dataset)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], tasks=("bbox",), distributed=False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    inference_on_dataset(model, val_loader, evaluator)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a Detectron2 model")
    parser.add_argument("--dataset", choices=["teeth3ds_coco36", "teeth3ds_coco"], default="teeth3ds_coco36")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--infer", action="store_true", help="Run inference")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    args = parser.parse_args()

    # Register the datasets
    if args.dataset == "teeth3ds_coco36":
        register_coco_instances("teeth3ds_coco36_train", {}, "./datasets/teeth3ds/teeth3ds_coco36/train_annotation.json", "./datasets/teeth3ds/teeth3ds_coco36/train")
        register_coco_instances("teeth3ds_coco36_test", {}, "./datasets/teeth3ds/teeth3ds_coco36/test_annotation.json", "./datasets/teeth3ds/teeth3ds_coco36/test")
    elif args.dataset == "teeth3ds_coco":
        register_coco_instances("teeth3ds_coco_train", {}, "./datasets/teeth3ds/teeth3ds_coco/train_annotation.json", "./datasets/teeth3ds/teeth3ds_coco/train")
        register_coco_instances("teeth3ds_coco_test", {}, "./datasets/teeth3ds/teeth3ds_coco/test_annotation.json", "./datasets/teeth3ds/teeth3ds_coco/test")

    if args.train:
       train(args)
    if args.infer:
       infer(args)
    if args.eval:
       eval(args)