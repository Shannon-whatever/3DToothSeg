import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


from models.main_network import ToothSegNet
from dataset.teeth3ds import Teeth3DSDataset
from utils.other_utils import output_pred_ply, save_metrics_to_txt
from utils.color_utils import label2color_lower, label2color_upper
from utils.metric_utils import calculate_miou
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
        
        miou = []
        per_class_iou = []
        merge_iou = []


        with torch.no_grad():
            loop_val = tqdm(dataloader, desc=f"Validating", leave=False)
            for batch_idx, batch_data in enumerate(loop_val):
                pointcloud = batch_data['pointclouds']
                point_coords = batch_data['point_coords']
                face_info = batch_data['face_infos']
                file_name = batch_data['file_names']
                labels = batch_data['labels']

                renders = batch_data['renders'].to(args.device)
                cameras_Rt = batch_data['cameras_Rt'].to(args.device)
                cameras_K = batch_data['cameras_K'].to(args.device)


                pointcloud = pointcloud.to(args.device)
                labels = labels.to(args.device)
                _, _, point_seg_result, _ = model(pointcloud, renders, cameras_Rt, cameras_K) # (B, num_classes, N_pc)
                # pred_softmax = torch.nn.functional.softmax(point_seg_result, dim=1)
                # _, pred_classes = torch.max(pred_softmax, dim=1) # (B, N_pc)

                # calculate miou
                pred_classes = point_seg_result.argmax(dim=1)  # (B, N_pc)
                # pred_classes[pred_classes == 17] = 0
                # pred_classes[pred_classes == 18] = 0
                
                miou_batch, per_class_iou_batch, merge_iou_batch = calculate_miou(
                    pred_classes, labels, n_class=17)
                miou.append(miou_batch)
                per_class_iou.append(per_class_iou_batch)
                merge_iou.append(merge_iou_batch)

                if save_result:
                    pred_classes = pred_classes.cpu().numpy().astype(np.uint8)

                    bs = len(point_coords)
                    for i in range(bs):
                        file_name_i = file_name[i]
                        if 'upper' in file_name_i:
                            save_path = os.path.join(args.save_predict_mask_dir, 'upper', f"{file_name_i}_predict.ply")
                            pred_mask = upper_palette[pred_classes[i]]
                        elif 'lower' in file_name_i:
                            save_path = os.path.join(args.save_predict_mask_dir, 'lower', f"{file_name_i}_predict.ply")
                            pred_mask = lower_palette[pred_classes[i]]
                        output_pred_ply(pred_mask, None, save_path, point_coords[i], face_info[i])


                    print(f"Predict end, visual result saved at {args.save_predict_mask_dir}")
                
                loop_val.set_postfix(miou=miou_batch)
            

            miou = torch.stack(miou)
            miou = miou.mean().item()

            per_class_iou = torch.stack(per_class_iou).mean(dim=0) # (C, )
            merge_iou = torch.stack(merge_iou).mean(dim=0) # (num_pairs, )


            # save metrics
            save_metrics_to_txt(
                filepath=os.path.join(args.save_dir, f"metrics_{args.provide_files}_miou{miou:.3f}.txt"),
                miou=miou,
                per_class_miou=per_class_iou.cpu().numpy(),
                merge_iou=merge_iou.cpu().numpy(),
                )
            
            print(f"Evaluation mIoU: {miou:.4f} for {args.provide_files}. Detail metrics saved to {args.save_dir}")


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
                dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate_fn
            )


    model = ToothSegNet(in_channels=6, num_classes=17, use_pretrain=args.load_ckp).to(args.device)

    predict(dataloader, model, args, save_result=args.save_visual)
