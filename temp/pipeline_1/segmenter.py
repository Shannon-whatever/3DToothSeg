import torch
from efficient_sam.efficient_sam import build_efficient_sam

class EfficientSAMSegmenter:
    def __init__(self, cfg):
        self.device = cfg.MODEL.DEVICE

        self.model = build_efficient_sam(
            encoder_patch_embed_dim=192,
            encoder_num_heads=3,
            checkpoint="./ckpts/efficient_sam_ckpts/efficient_sam_vitt.pt",
        )

    def __call__(
            self, 
            batched_images: torch.Tensor, 
            batched_points: torch.Tensor = None, 
            batched_point_labels: torch.Tensor = None):
        self.model.to(self.device)
        self.model.eval()
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            masks = self.model(batched_images, batched_points, batched_point_labels)
            # Arguments:
            # batched_images: A tensor of shape [B, 3, H, W]
            # batched_points: A tensor of shape [B, num_queries, max_num_pts, 2]
            # batched_point_labels: A tensor of shape [B, num_queries, max_num_pts]
            # Returns:
            # A tuple of two tensors:
            # low_res_mask: A tensor of shape [B, max_num_queries, 256, 256] of predicted masks
            # iou_predictions: A tensor of shape [B, max_num_queries] of estimated IOU scores
        return masks