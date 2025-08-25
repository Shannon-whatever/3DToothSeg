import torch

def boxes_to_points_labels(boxes):
    """
    Convert [N,4] boxes to batched_points and batched_point_labels tensors for EfficientSAM
    Keeps outputs on the same device as boxes
    """
    device = boxes.device
    num_boxes = boxes.shape[0]

    points_list = []
    labels_list = []

    for i in range(num_boxes):
        x1, y1, x2, y2 = boxes[i]
        points_list.append([[x1, y1], [x2, y2]])    # 2 points per box
        labels_list.append([2, 3])                  # 2=top-left, 3=bottom-right

    batched_points = torch.tensor(points_list, device=device).unsqueeze(0)   # [B=1, num_queries=num_boxes, num_pts=2, 2]
    batched_point_labels = torch.tensor(labels_list, device=device).unsqueeze(0)  # [B=1, num_queries=num_boxes, num_pts=2]

    return batched_points, batched_point_labels