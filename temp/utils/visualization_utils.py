import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)  # RGBA
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])  # RGBA

    h, w = mask.shape[-2:]
    mask = mask.astype(bool)  # ensure binary

    overlay = np.zeros((h, w, 4))
    overlay[mask] = color  # apply only where mask==1

    ax.imshow(overlay)

def show_box(box, ax, color="red", linewidth=2):
    """Draw a bounding box [x1, y1, x2, y2] on matplotlib axis."""
    x1, y1, x2, y2 = box
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=linewidth, edgecolor=color, facecolor="none"
    )
    ax.add_patch(rect)

def show_all_mask(batch, outputs, save_dir, sample_idx=0, fname_prefix=None):
    """
    Save visualization of predictions for one sample in a batch.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Image
    image = batch[sample_idx]["image"].permute(1,2,0).cpu().numpy().astype("uint8")

    # Predictions
    masks = outputs["masks"][sample_idx]       # [N, H, W]
    # boxes = outputs["boxes"]                   # [N, 4]
    # classes = outputs["classes"]               # [N]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.axis("off")

    # Draw each prediction
    for i in range(masks.shape[0]):
        # Draw bounding box
        # show_box(boxes[i], ax, color="red")

        # Draw mask (binary or soft depending on your pipeline)
        show_mask(masks[i], ax, random_color=True)

    # title = f"Pred masks + boxes (classes={classes.tolist()})"
    # ax.set_title(title)

    # File name
    if fname_prefix is None:
        fname_prefix = os.path.splitext(os.path.basename(batch[sample_idx]["file_name"]))[0]
    save_path = os.path.join(save_dir, f"{fname_prefix}_pred.png")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {save_path}")