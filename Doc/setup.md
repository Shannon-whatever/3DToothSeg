
cd models/PointTransformer/libs/pointops
python setup.py install



ensure gcc version < 11

conda install gcc_linux-64=7.3.0
conda install gxx_linux-64=7.3.0
 
cd ~/anaconda3/envs/UPSNet-918/bin
 
ln -s x86_64-conda_cos6-linux-gnu-gcc gcc
ln -s x86_64-conda_cos6-linux-gnu-g++ g++