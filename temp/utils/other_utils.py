"""
Copied from https://github.com/Shannon-whatever/3DToothSeg/blob/main/utils/other_utils.py
"""
def save_metrics_to_txt(filepath, num_classes, miou, biou, per_class_miou, merge_iou, 
                        merge_pairs=[(1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15), (8, 16)]):
    """
    Save evaluation metrics to a plain text (.txt) file.

    Args:
        filepath: str, path to save the txt file
        miou: float, overall mean IoU
        per_class_miou: Tensor (C,)
        merge_iou: Tensor (num_pairs,)
        class_names: Optional[List[str]], names for each class
        merge_names: Optional[List[str]], names for each merge pair
    """
    class_names = [f"Class {i}" for i in range(num_classes)]
    merge_names = [f"T{a}/T{b}" for a, b in merge_pairs]
    with open(filepath, "w") as f:
        f.write("==== Segmentation Evaluation Metrics ====\n\n")
        f.write(f"Overall mIoU: {miou:.4f}\n\n")
        f.write(f"Overall bIoU: {biou:.4f}\n\n")

        f.write("Per-Class mIoU:\n")
        for i, iou in enumerate(per_class_miou):
            name = class_names[i] if class_names else f"Class {i}"
            f.write(f"  {name:<10s}: {iou:.4f}\n")
        f.write("\n")

        if merge_iou is not None:
            f.write("Merged-Class IoU:\n")
            for i, iou in enumerate(merge_iou):
                name = merge_names[i] if merge_names else f"Pair {i}"
                f.write(f"  {name:<10s}: {iou:.4f}\n")