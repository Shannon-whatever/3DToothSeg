# 3DToothSeg Setup Guide

## Env Setup

### 1. Clone the Repository

```bash
git clone https://github.com/zykev/3DToothSeg.git
```

### 2. Install PyTorch and Torch-geometric(CUDA 11.8)

```bash
conda create -n 3dtooth python=3.10

conda activate 3dtooth

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch-geometric -f https://data.pyg.org/whl/torch-2.4.0+cu118.html

```

<!-- ### 3. Install PyTorch3D

```bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
``` -->

### 3. Install Other Requirements

```bash
pip install -r requirements.txt
```

### 4. Install pointops

```bash
cd models/PointTransformer/libs/pointops
python setup.py install
```

Remark: Ensure gcc version < 11 before installing pointops. Otherwise, install gcc version under 3dtooth env by:

```bash
conda install -c conda-forge gcc_linux-64=9.4.0

conda install -c conda-forge gxx_linux-64=9.4.0

 
cd ~/anaconda3/envs/3dtooth/bin (change to your anaconda/miniconda path)
 
ln -s x86_64-conda_cos6-linux-gnu-gcc gcc
ln -s x86_64-conda_cos6-linux-gnu-g++ g++
```

Check gcc and g++ version by:
```bash
gcc --version
g++ --version
```



## Pretrained model

PSPNet checkpoint:
   - [Download checkpoints from this repo and rename them by the following format](https://drive.google.com/drive/folders/15wx9vOM0euyizq-M1uINgN0_wjVRf9J3)

   - Organize like: 
   ```
.checkpoints/
└── PSPNet/
    ├── init_resnet50_v2.pth
    ├── train_ade20k_pspnet_epoch_100.pth
       
```

## Datasets

### Teeth3ds (https://crns-smartvision.github.io/teeth3ds/)

1. Download the following:

   - [Teeth3ds dataset by MICCAI challenge](https://osf.io/xctdy/)


2. Combine all the data parts into a single pair of upper and lower folders and organize them like this:

```
.datasets/
└── teeth3ds/
    ├── upper/
    │   └── xxxx(data id)
        └── xxxx ...
    ├── lower/
    │   └── xxxx(data id)
        └── xxxx ...

```

3. Download dataset split and extract all the txt files into one folder named split under .datasets/teeth3ds/

https://github.com/abenhamadou/3DTeethSeg_MICCAI_Challenges/tree/main/dataset 










