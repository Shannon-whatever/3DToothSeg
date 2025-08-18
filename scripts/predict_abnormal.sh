#!/bin/bash

# 指定使用哪个 GPU（比如使用第 1 张卡，编号为 0）
export CUDA_VISIBLE_DEVICES=0

# 启动 Python 脚本，传入自定义参数
python predict.py \
    --data_dir .datasets/teeth3ds \
    --num_points 16000 \
    --sample_points 16000 \
    --sample_views 10 \
    --batch_size 4 \
    --save_dir exp/baseline_reproduce/abnormal_eg \
    --device cuda:0 \
    --num_workers 4 \
    --provide_files exp/baseline_reproduce/abnormal_examples.txt \
    --load_ckp exp/baseline_reproduce/toothseg_epoch100_miou0.938.pth \
    --save_visual \

