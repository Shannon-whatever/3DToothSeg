import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from einops import rearrange
from torch.utils.data import DataLoader


from models.main_network import ToothSegNet
from dataset.teeth3ds import Teeth3DSDataset
from utils.other_utils import output_pred_ply, output_pred_images, save_metrics_to_txt
from utils.color_utils import label2color_lower, label2color_upper
from utils.metric_utils import calculate_miou, calculate_miou_2d
from utils.dataset_utils import custom_collate_fn


def predict(dataloader, model, args, save_result=False):


        model.eval()

        lower_palette = np.array(
        [[label2color_lower[label][2][0],
          label2color_lower[label][2][1],
          label2color_lower[label][2][2]]
         for label in range(0, 17)], dtype=np.uint8) # (17, 3)
        
        upper_palette = np.array(
        [[label2color_upper[label][2][0],
          label2color_upper[label][2][1],
          label2color_upper[label][2][2]]
         for label in range(0, 17)], dtype=np.uint8)
        
        lower_palette_2d = np.concatenate([lower_palette, np.array([[0, 0, 0]], dtype=np.uint8)], axis=0) # (18, 3)
        upper_palette_2d = np.concatenate([upper_palette, np.array([[0, 0, 0]], dtype=np.uint8)], axis=0) # (18, 3)
        
        miou = []
        per_class_iou = []
        merge_iou = []

        miou_2d = []
        per_class_iou_2d = []


        with torch.no_grad():
            loop_val = tqdm(dataloader, desc=f"Validating", leave=False)
            for batch_idx, batch_data in enumerate(loop_val):
                pointcloud = batch_data['pointclouds']
                point_coords = batch_data['point_coords']
                face_info = batch_data['face_infos']
                file_name = batch_data['file_names']
                labels = batch_data['labels']

                renders = batch_data['renders'].to(args.device)
                masks = batch_data['masks'].to(args.device)
                cameras_Rt = batch_data['cameras_Rt'].to(args.device)
                cameras_K = batch_data['cameras_K'].to(args.device)


                pointcloud = pointcloud.to(args.device)
                labels = labels.to(args.device)
                predict_2d_masks, _, point_seg_result, _ = model(pointcloud, renders, cameras_Rt, cameras_K) 
                # point_seg_result: (B, num_classes=17, N_pc) predict_2d_masks: (B*N_v, 17+1, H, W)
                # 2d seg result
                pred_classes_2d = predict_2d_masks.argmax(dim=1)  # (B*N_v, H, W)
                masks = rearrange(masks, 'b nv h w -> (b nv) h w') # (B, N_v, H, W) -> (B*N_v, H, W)
                miou_batch_2d, per_class_iou_batch_2d = calculate_miou_2d(
                    pred_classes_2d, masks, n_class=17+1)
                miou_2d.append(miou_batch_2d)
                per_class_iou_2d.append(per_class_iou_batch_2d)


                # calculate miou for 3d seg
                pred_classes = point_seg_result.argmax(dim=1)  # (B, N_pc)
                # pred_classes[pred_classes == 17] = 0
                # pred_classes[pred_classes == 18] = 0
                
                miou_batch, per_class_iou_batch, merge_iou_batch = calculate_miou(
                    pred_classes, labels, n_class=17)
                miou.append(miou_batch)
                per_class_iou.append(per_class_iou_batch)
                merge_iou.append(merge_iou_batch)

                if save_result:
                    # visualization on 3d mesh and 2d masks
                    bs = len(point_coords)

                    pred_classes = pred_classes.cpu().numpy().astype(np.uint8)
                    pred_classes_2d = pred_classes_2d.cpu().numpy().astype(np.uint8)
                    pred_classes_2d = pred_classes_2d.reshape(bs, -1, *pred_classes_2d.shape[1:]) # (B, N_v, H, W)
                    masks = masks.cpu().numpy().astype(np.uint8)
                    masks = masks.reshape(bs, -1, *masks.shape[1:]) # (B, N_v, H, W)

                    # sample 5 views every 8 steps from masks and pred_classes_2d for visualization
                    sampled_indices = np.linspace(0, 40, 5, dtype=int)
                    pred_classes_2d_sampled = pred_classes_2d[:, sampled_indices, ...]
                    masks_sampled = masks[:, sampled_indices, ...]

                    for i in range(bs):
                        file_name_i = file_name[i]
                        file_view = file_name_i.split('_')[1]
                        file_id = file_name_i.split('_')[0]
                        save_dir = os.path.join(args.save_predict_mask_dir, file_view, file_id)
                        os.makedirs(save_dir, exist_ok=True)

                        if file_view == 'upper':
                            pred_mask = upper_palette[pred_classes[i]]
                            pred_mask_2d = upper_palette_2d[pred_classes_2d_sampled[i]] # (N_v, H, W, 3)
                            gt_mask_2d = upper_palette_2d[masks_sampled[i]] # (N_v, H, W, 3)
                        elif file_view == 'lower':
                            pred_mask = lower_palette[pred_classes[i]]
                            pred_mask_2d = lower_palette_2d[pred_classes_2d_sampled[i]]
                            gt_mask_2d = lower_palette_2d[masks_sampled[i]]
                        save_path_3d = os.path.join(save_dir, f"{file_name_i}_predict.ply")
                        output_pred_ply(pred_mask, None, save_path_3d, point_coords[i], face_info[i])
                        output_pred_images(pred_mask_2d, gt_mask_2d, save_dir, file_name_i)

                loop_val.set_postfix(miou=miou_batch)
            
                
            
            # 3d mIoU
            miou = torch.stack(miou).mean().item()
            per_class_iou = torch.stack(per_class_iou).mean(dim=0) # (C, )
            merge_iou = torch.stack(merge_iou).mean(dim=0) # (num_pairs, )

            # 2d mIoU
            miou_2d = torch.stack(miou_2d).mean().item()
            per_class_iou_2d = torch.stack(per_class_iou_2d).mean(dim=0) # (C+1, )


            # save metrics
            save_name = os.path.splitext(os.path.basename(args.provide_files))[0]
            save_metrics_to_txt(
                filepath=os.path.join(args.save_dir, f"metrics_{save_name}_miou{miou:.3f}.txt"),
                num_classes=17,
                miou=miou,
                per_class_miou=per_class_iou.cpu().numpy(),
                merge_iou=merge_iou.cpu().numpy(),
                )
            
            save_metrics_to_txt(
                filepath=os.path.join(args.save_dir, f"metrics_2d_{save_name}_miou{miou_2d:.3f}.txt"),
                num_classes=18,
                miou=miou_2d,
                per_class_miou=per_class_iou_2d.cpu().numpy(),
                merge_iou=None,
                )
            
            print(f"Evaluation mIoU: {miou:.4f} for {save_name}. Detail metrics saved to {args.save_dir}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.datasets/teeth3ds')
    parser.add_argument('--num_points', type=int, default=16000)
    parser.add_argument('--sample_points', type=int, default=16000)
    parser.add_argument('--sample_views', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='exp/train')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--provide_files', type=str, default=None, help='Provide txt files for testing')
    parser.add_argument('--load_ckp', type=str, default=None, help='Use trained checkpoint path')
    parser.add_argument('--save_visual', action='store_true', help='Save visual results')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_visual:
        args.save_predict_mask_dir = os.path.join(args.save_dir, 'predict_masks')
        os.makedirs(os.path.join(args.save_predict_mask_dir, 'upper'), exist_ok=True)
        os.makedirs(os.path.join(args.save_predict_mask_dir, 'lower'), exist_ok=True)



    dataset = Teeth3DSDataset(
                root=args.data_dir, in_memory=False,
                force_process=False, train_test_split=0, is_train=False,
                num_points=16000, sample_points=16000, sample_views=args.sample_views,
                provide_files=args.provide_files
            )

    dataloader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate_fn
            )


    model = ToothSegNet(in_channels=6, num_classes=17, use_pretrain=args.load_ckp).to(args.device)

    predict(dataloader, model, args, save_result=args.save_visual)
