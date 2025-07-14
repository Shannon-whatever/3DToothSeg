import os
import glob
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from models.PTv1.point_transformer_seg import PointTransformerSeg38
from dataset.data1 import ToothData
import numpy as np
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/process/train_downsampled')
    parser.add_argument('--num_points', type=int, default=16000)
    parser.add_argument('--sample_points', type=int, default=16000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_1')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()

def train():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    file_list = glob.glob(os.path.join(args.data_dir, '*.ply'))
    print(f"Found {len(file_list)} ply files in {args.data_dir}")

    dataset = ToothData(args, file_list, with_label=True)
    print(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )

    model = PointTransformerSeg38(
        in_channels=6, num_classes=17, pretrain=False, enable_pic_feat=False
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs(args.save_dir, exist_ok=True)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch_idx, (pointcloud, labels, face_info) in enumerate(dataloader):
            
            pointcloud = pointcloud.to(device).permute(0, 2, 1).contiguous()  # (B, 6, N)
            labels = labels.to(device)  # (B, N)

            optimizer.zero_grad()
            outputs, _ = model(pointcloud)  # outputs: (B, num_classes, N)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # pred_classes = outputs.argmax(dim=1)  # (B, N)

            # unique_labels = np.unique(labels.cpu().numpy())
            # unique_preds = np.unique(pred_classes.detach().cpu().numpy())
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Average Loss: {avg_loss:.4f}")
        # print(f"  Unique True Labels: {unique_labels}")
        # print(f"  Unique Predicted Labels: {unique_preds}")
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            save_path = os.path.join(args.save_dir, f"point_transformer_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model checkpoint to {save_path}")

if __name__ == '__main__':
    train()