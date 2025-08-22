import torch 
import argparse
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid")

from pipeline_1 import DetectionSegmentationPipeline

from detectron2.data import build_detection_test_loader 

from utils import (
    setup_cfg,
    masks_to_label_map_batch, 
    color_mask_to_label_maps_batch, 
    calculate_miou_2d, 
    save_metrics_to_txt
)


def predict(args, cfg):
    miou = []
    per_class_iou = []
    
    with torch.no_grad(): 
        for idx, batch in enumerate(tqdm(test_dataloader, desc="Testing", unit="batch")):
            # batch: list of dict of len = batch_size
            # dict keys: file_name, height, width, image_id, image
            # Example:
            # file_name: "./datasets/teeth3ds/teeth3ds_coco36/test/4MC4KRQV_upper_0.png"
            # height, width: 1024 1024
            # image: torch.Size([3, 1024, 1024])
            outputs = pipeline(batch)

            gt_labels = color_mask_to_label_maps_batch(batch, device=cfg.MODEL.DEVICE)
            pred_labels = masks_to_label_map_batch(
                outputs["masks"], outputs["classes"], outputs["valid_mask"]
            )

            miou_batch, per_class_iou_batch = calculate_miou_2d(pred_labels, gt_labels, n_class=pipeline.cfg.MODEL.ROI_HEADS.NUM_CLASSES+1)
            # print(f"miou_batch: {miou_batch}, per_class_iou_batch: {per_class_iou_batch}")

            miou.append(miou_batch)
            per_class_iou.append(per_class_iou_batch)

        miou = torch.stack(miou).mean().item()
        per_class_iou = torch.stack(per_class_iou).mean(dim=0) 

        if args.save_metrics:
            save_path = os.path.join(cfg.OUTPUT_DIR, "miou_results.txt")
            save_metrics_to_txt(save_path, 
                                pipeline.cfg.MODEL.ROI_HEADS.NUM_CLASSES, 
                                miou.cpu().numpy(), 
                                per_class_iou.cpu().numpy())


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_metrics', action='store_true', help='Save visual results')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    cfg = setup_cfg("infer")
    pipeline = DetectionSegmentationPipeline(cfg)
    test_dataloader = build_detection_test_loader(cfg, dataset_name=cfg.DATASETS.TEST[0], batch_size=16)

    predict(args, cfg)