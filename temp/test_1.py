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
    masks_to_label_maps_batch, 
    color_masks_to_label_maps_batch, 
    calculate_miou_2d, 
    save_metrics_to_txt,
    show_all_mask,
)


def predict(args, cfg):
    miou = []
    per_class_iou = []
    
    with torch.no_grad(): 
        for idx, batch in enumerate(tqdm(test_dataloader, desc="Testing", unit="batch")):
            outputs = pipeline(batch)
            # print(outputs["masks"].shape, outputs["boxes"].shape, outputs["box_scores"].shape, outputs["classes"].shape)

            gt_labels = color_masks_to_label_maps_batch(batch, device=cfg.MODEL.DEVICE)
            pred_labels = masks_to_label_maps_batch(outputs["masks"], outputs["classes"])
            # print(f"unique gt_labels: {gt_labels.unique()}, unique pred_labels: {pred_labels.unique()}")

            miou_batch, per_class_iou_batch = calculate_miou_2d(pred_labels, gt_labels, n_class=pipeline.cfg.MODEL.ROI_HEADS.NUM_CLASSES+1)
            # print(f"miou_batch: {miou_batch}, per_class_iou_batch: {per_class_iou_batch}")

            miou.append(miou_batch)
            per_class_iou.append(per_class_iou_batch)


        miou = torch.stack(miou).mean()
        per_class_iou = torch.stack(per_class_iou).mean(dim=0) 

        if args.save_metrics:
            try:
                save_path = os.path.join(cfg.OUTPUT_DIR, f"miou_results_{cfg.DATASETS.TEST[0]}.txt")
                save_metrics_to_txt(filepath=save_path, 
                                    num_classes=pipeline.cfg.MODEL.ROI_HEADS.NUM_CLASSES+1, 
                                    miou=miou.cpu().numpy(), 
                                    biou=None,
                                    per_class_miou=per_class_iou.cpu().numpy(),
                                    merge_iou=None)
            except Exception as e:
                print(f"Error saving metrics: {e}")
                torch.save({"miou": miou, "per_class_iou": per_class_iou}, "temp_metrics.pt")
            finally:
                pass

        if args.visualize_masks:
            try:
                save_dir = "/home/zychen/Documents/Project_shno/3DToothSeg/temp/output_vis"
                show_all_mask(batch, outputs, save_dir=save_dir, sample_idx=0)
            except Exception as e:
                print(f"Error visualizing masks: {e}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_metrics', action='store_true', help='Save metrics results')
    parser.add_argument('--visualize_masks', action='store_true', help='Save visual results')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    cfg = setup_cfg("infer")
    pipeline = DetectionSegmentationPipeline(cfg)
    test_dataloader = build_detection_test_loader(cfg, dataset_name=cfg.DATASETS.TEST[0], batch_size=1)

    predict(args, cfg)