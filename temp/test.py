from pipeline import DetectionSegmentationPipeline
import warnings
import torch
import cv2
from utils import visualize_and_save

warnings.filterwarnings("ignore", message=".*torch.meshgrid:.*")

model = DetectionSegmentationPipeline()
img = cv2.imread("./sample/4MC4KRQV_upper_0.png")
outputs = model.run(img)
print("Boxes shape:", outputs["boxes"].shape)
print("Masks shape:", outputs["masks"].shape)
print("Scores shape:", outputs["box_scores"].shape)
print("Classes shape:", outputs["classes"].shape)

visualize_and_save(img, outputs, save_dir="./output_vis", sample_idx=0, fname_prefix="test")