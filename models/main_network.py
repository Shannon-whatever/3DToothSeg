import torch
import torch.nn as nn

from models.PTv1.point_transformer_seg import PointTransformerSeg38
from models.Seg2d.pspnet import PSPNet


class ToothSegNet(nn.Module):
    def __init__(self, in_channels=6, num_classes=17, 
                 use_pretrain_3d='models/PTv1/point_best_model.pth', 
                 use_pretrain_2d='.checkpoints/PSPNet/train_ade20k_pspnet50_epoch_100.pth'):
        super().__init__()

        self.seg_model_3d = PointTransformerSeg38(
            in_channels=in_channels, num_classes=num_classes,
            pretrain=False, enable_pic_feat=False
        )

        self.seg_model_2d = PSPNet(layers=50, classes=num_classes, zoom_factor=8, use_ppm=True, pretrained=True)


        if use_pretrain_3d is not None:
            print(f"Loading 3d pretrained model from {use_pretrain_3d}")
            pretrained_model = torch.load(use_pretrain_3d)
            self.seg_model_3d.load_state_dict(pretrained_model)

        if use_pretrain_2d is not None:
            print(f"Loading 2d pretrained model from {use_pretrain_2d}")
            pretrained_model = torch.load(use_pretrain_2d)
            pretrained_weights = pretrained_model['state_dict']
            pretrained_weights = {k.replace('module.', ''): v for k, v in pretrained_weights.items()}
            # 过滤掉和分类头相关的权重（cls 和 aux）
            keys_to_remove = [k for k in pretrained_weights.keys() if k.startswith('cls.4') or k.startswith('aux.4')]
            for k in keys_to_remove:
                pretrained_weights.pop(k)

            self.seg_model_2d.load_state_dict(pretrained_weights, strict=False)

    def forward(self, pointcloud, renders):

        pointcloud = pointcloud.permute(0, 2, 1).contiguous()
        renders = renders.reshape(-1, renders.shape[2], renders.shape[3], renders.shape[4])

        predict_2d_masks = self.seg_model_2d(renders)
        predict_pc_labels, _ = self.seg_model_3d(pointcloud)

        return predict_2d_masks, predict_pc_labels