import torch

def boxes_to_points_labels(batched_boxes):
    """
    Convert [B, N, 4] boxes to [B, N, 2, 2] points and [B, N, 2] labels for EfficientSAM
    Args:
        batched_boxes: torch.Tensor [B, N, 4] -> (x1, y1, x2, y2)
    Returns:
        batched_points: torch.Tensor [B, N, 2, 2]
        batched_labels: torch.Tensor [B, N, 2]
    """
    B, N, _ = batched_boxes.shape

    # Split coordinates
    x1 = batched_boxes[..., 0]
    y1 = batched_boxes[..., 1]
    x2 = batched_boxes[..., 2]
    y2 = batched_boxes[..., 3]

    # Construct points: [B, N, 2, 2]
    batched_points = torch.stack([
        torch.stack([x1, y1], dim=-1),  # top-left
        torch.stack([x2, y2], dim=-1)   # bottom-right
    ], dim=-2)

    # Construct labels: [B, N, 2], always [2, 3]
    batched_labels = torch.tensor([2, 3], device=batched_boxes.device)
    batched_labels = batched_labels.expand(B, N, -1)

    return batched_points, batched_labels