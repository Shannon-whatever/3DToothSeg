from pipeline import DetectionSegmentationPipeline
import warnings
import torch
import cv2

warnings.filterwarnings("ignore", message=".*torch.meshgrid:.*")

model = DetectionSegmentationPipeline()
img = cv2.imread("./sample/4MC4KRQV_upper_0.png")
masks, boxes, scores, classes = model.run(img)
print("Boxes shape:", boxes.shape)
print("Masks shape:", masks[0].shape)
print("Scores shape:", scores.shape)
print("Classes shape:", classes.shape)