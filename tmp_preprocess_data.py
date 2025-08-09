# %%
import os
import matplotlib.pyplot as plt
from PIL import Image

def visualize_comparisons(root_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for category in ["upper", "lower"]:
        category_dir = os.path.join(root_dir, category)
        if not os.path.exists(category_dir):
            continue
        
        for sample_id in os.listdir(category_dir):
            sample_dir = os.path.join(category_dir, sample_id)
            gt_dir = os.path.join(sample_dir, "gt_mask")
            pred_dir = os.path.join(sample_dir, "pred_mask")
            if not (os.path.exists(gt_dir) and os.path.exists(pred_dir)):
                continue
            
            gt_files = sorted(os.listdir(gt_dir))
            pred_files = sorted(os.listdir(pred_dir))
            
            # 匹配相同视角
            paired_files = [(gt, pred) for gt, pred in zip(gt_files, pred_files) if os.path.splitext(gt)[0] == os.path.splitext(pred)[0]]
            if not paired_files:
                continue
            
            fig, axes = plt.subplots(len(paired_files), 2, figsize=(6, 3 * len(paired_files)))
            if len(paired_files) == 1:
                axes = [axes]  # 单行处理
            
            for idx, (gt_file, pred_file) in enumerate(paired_files):
                gt_img = Image.open(os.path.join(gt_dir, gt_file)).convert("RGB")
                pred_img = Image.open(os.path.join(pred_dir, pred_file)).convert("RGB")
                
                axes[idx][0].imshow(gt_img)
                axes[idx][0].set_title("Ground Truth")
                axes[idx][0].axis("off")
                
                axes[idx][1].imshow(pred_img)
                axes[idx][1].set_title("Prediction")
                axes[idx][1].axis("off")
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"{category}_{sample_id}.png")
            plt.savefig(save_path, dpi=300)
            plt.close(fig)

# 使用示例
root_folder = "exp/baseline_reproduce/predict_masks"
save_folder = "exp/baseline_reproduce/comparison_results"
visualize_comparisons(root_folder, save_folder)

# %%
