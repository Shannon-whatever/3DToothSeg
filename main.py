import os
import glob
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb

from models.PTv1.point_transformer_seg import PointTransformerSeg38
from dataset.teeth3ds import Teeth3DSDataset
from utils.other_utils import output_pred_ply, label2color_lower


class ToothSegmentationPipeline:
    def __init__(self, args):
        self.args = args
        self.device = self.args.device

    def get_dataloader(self):
        upper_files = glob.glob(os.path.join(self.args.data_dir, 'upper', '*.ply'))
        lower_files = glob.glob(os.path.join(self.args.data_dir, 'lower', '*.ply'))
        file_list = upper_files + lower_files
        print(f"Found {len(file_list)} ply files in {self.args.data_dir}")

        train_dataset = Teeth3DSDataset(
            root=self.args.data_dir, in_memory=False,
            force_process=True, train_test_split=self.args.train_test_split, is_train=True,
            num_points=self.args.num_points, sample_points=self.args.sample_points
        )
        test_dataset = Teeth3DSDataset(
            root=self.args.data_dir, in_memory=False,
            force_process=True, train_test_split=self.args.train_test_split, is_train=False,
            num_points=self.args.num_points, sample_points=self.args.sample_points
        )

        print(f"Dataset size: Train: {len(train_dataset)}, Test: {len(test_dataset)}")

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.num_workers, pin_memory=True, collate_fn=self.custom_collate_fn
        )

        test_dataloader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.num_workers, pin_memory=True, collate_fn=self.custom_collate_fn
        )
        return train_dataloader, test_dataloader


    def build_model(self, num_classes):
        model = PointTransformerSeg38(
            in_channels=6, num_classes=num_classes,
            pretrain=False, enable_pic_feat=False
        ).to(self.device)
        return model

    def train(self, train_dataloader, test_dataloader, model, log=None):

        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=1e-6)
        criterion = torch.nn.CrossEntropyLoss()
        

        model.train()
        for epoch in range(self.args.epochs):
            total_loss = 0.0
            loop = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{self.args.epochs}]", leave=False)

            for batch_idx, (pointcloud, labels, point_coords, face_info, renders, masks, file_name) in enumerate(loop):
                pointcloud = pointcloud.to(self.device).permute(0, 2, 1).contiguous()
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs, _ = model(pointcloud)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                loop.set_postfix(loss=loss.item())

                if log:
                    wandb_run.log({"batch_loss": loss.item(), "step": epoch * len(train_dataloader) + batch_idx})

            avg_loss = total_loss / len(train_dataloader)
            tqdm.write(f"Epoch [{epoch+1}/{self.args.epochs}] - Average Loss: {avg_loss:.4f}")
            scheduler.step()

            if log:
                log.log({"epoch": epoch + 1, "epoch_loss": avg_loss, "lr": optimizer.param_groups[0]['lr']})


            if (epoch + 1) % self.args.eval_epoch_step == 0 or (epoch + 1) == self.args.epochs:

                self.predict(test_dataloader, model, epoch, log=log)

                save_path = os.path.join(self.args.save_dir, f"point_transformer_epoch{epoch+1}.pth")
                torch.save(model.state_dict(), save_path)
                print(f"Saved model checkpoint to {save_path}")

    def predict(self, dataloader, model, current_epoch=None, log=None, use_pretrained='models/PTv1/point_best_model.pth'):
 
        if use_pretrained is not None:
            pretrained_dict = torch.load(use_pretrained)
            print(f"Loading pretrained model from {use_pretrained}")
            model.load_state_dict(pretrained_dict)

        model.eval()

        lower_palette = np.array(
            [[125, 125, 125]] +
            [[label2color_lower[label][2][0],
              label2color_lower[label][2][1],
              label2color_lower[label][2][2]]
             for label in range(1, 17)], dtype=np.uint8)

        # case by case evaluation
        # pointcloud, point_coords, face_info = dataloader.dataset.get_by_name(self.args.case)
        # pointcloud = pointcloud.unsqueeze(0).to(self.device)
        # point_coords = point_coords[None, :]
        # face_info = face_info[None, :]
        # pointcloud = pointcloud.permute(0, 2, 1).contiguous()

        with torch.no_grad():
            for pointcloud, labels, point_coords, face_info, renders, masks, file_name in dataloader:
                pointcloud = pointcloud.to(self.device).permute(0, 2, 1).contiguous()
                point_seg_result, _ = model(pointcloud)
                pred_softmax = torch.nn.functional.softmax(point_seg_result, dim=1)
                _, pred_classes = torch.max(pred_softmax, dim=1)

                if use_pretrained or (current_epoch is not None and (current_epoch + 1) == self.args.epochs):
                    pred_mask = pred_classes.squeeze(0).cpu().numpy().astype(np.uint8)
                    pred_mask[pred_mask == 17] = 0
                    pred_mask[pred_mask == 18] = 0
                    pred_mask = lower_palette[pred_mask]

                    bs = len(point_coords)
                    for i in range(bs):
                        file_name_i = file_name[i]
                        if 'upper' in file_name_i:
                            save_path = os.path.join(self.args.save_predict_mask_dir, 'upper', file_name_i.replace('process', 'predict'))
                        elif 'lower' in file_name_i:
                            save_path = os.path.join(self.args.save_predict_mask_dir, 'lower', file_name_i.replace('process', 'predict'))
                        output_pred_ply(pred_mask[i], None, save_path, point_coords[i], face_info[i])


                    print(f"Predict end, result saved at {self.args.save_predict_mask_dir}")

    def custom_collate_fn(self, batch):
        pointclouds = []
        labels = []
        point_coords_list = []
        face_infos = []
        renders = []
        masks = []
        file_names = []

        for pc, label, p_coords, f_info, render, mask, file_name in batch:
            pointclouds.append(pc)
            labels.append(label)
            point_coords_list.append(p_coords)  # 不堆叠，保留为 list of np.array
            face_infos.append(f_info)
            renders.append(render)
            masks.append(mask)
            file_names.append(file_name)

        # 堆叠固定 shape 的数据
        pointclouds = torch.stack(pointclouds)  # (B, num_points, 6)
        labels = torch.stack(labels)            # (B, num_points)
        renders = torch.stack(renders)          # (B, N, C, H, W)
        masks = torch.stack(masks)              # (B, N, H, W)

        return pointclouds, labels, point_coords_list, face_infos, renders, masks, file_names
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.datasets/teeth3ds/sample')
    parser.add_argument('--num_points', type=int, default=16000)
    parser.add_argument('--sample_points', type=int, default=16000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='exp')
    parser.add_argument('--save_predict_mask_dir', type=str, default='.datasets/teeth3ds/sample/predict')
    parser.add_argument('--eval_epoch_step', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_test_split', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--pretrain_model_path', type=str, default='models/PTv1/point_best_model.pth')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_predict_mask_dir, 'upper'), exist_ok=True)
    os.makedirs(os.path.join(args.save_predict_mask_dir, 'lower'), exist_ok=True)
    
    if args.use_wandb:
        wandb_run = wandb.init(project='3dtooth_seg', name='3dtooth_seg', config=vars(args))
    else:
        wandb_run = None

    pipeline = ToothSegmentationPipeline(args)
    
    train_dataloader, test_dataloader = pipeline.get_dataloader()
    if args.train:
        print("Starting training...")
        model = pipeline.build_model(num_classes=17)
        pipeline.train(train_dataloader, test_dataloader, model, log=wandb_run)
    else:
        print("Starting prediction...")
        model = pipeline.build_model(num_classes=17 + 2)
        pipeline.predict(test_dataloader, model, log=wandb_run, use_pretrained=True)
    
    
    