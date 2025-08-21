from pipeline import DetectionSegmentationPipeline
import warnings
import torch
import cv2

warnings.filterwarnings("ignore", message=".*torch.meshgrid:.*")

model = DetectionSegmentationPipeline()
img = cv2.imread("/home/zychen/Documents/Project_shno/3DToothSeg/temp/0AAQ6BO3_upper_94.png")
masks, boxes, scores, classes = model.run(img)