# ########################### on HPC #########################################
# module purge
# module load CUDA/9.0.176 GCC/5.4.0-2.26 NCCL/2.3.7-CUDA-9.0.176

########################### build conda env ###################################
# conda init
# source ~/.bashrc
# conda create -n bpnp python=3.7 -y
python -m venv ~/.bpnp

########################### activate env ###################################
# eval "$(conda shell.bash hook)"
# conda activate bpnp
# which python
source ~/.bpnp/bin/activate

########################### install deps env ###################################
# conda install -y -c conda-forge ipython ninja cython matplotlib opencv=3.4 tqdm requests six scipy
# conda install -y -c pytorch     pytorch torchvision cudatoolkit=10.1
pip install kornia
pip install ninja cython matplotlib opencv tqdm requests six scipy
pip install pytorch torchvision
