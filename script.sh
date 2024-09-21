#!/bin/bash

#SBATCH --job-name=test
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err


PY_ARGS=${@:1}

# load virtual environment


module load cuda-12.1

# export PATH=/pkgs/anaconda3/bin:$PATH
# export PYTHONPATH=$HOME/condaenvs/pytorch-2.0:$PYTHONPATH
# export LD_LIBRARY_PATH=/pkgs/cuda-10.0/lib64:/pkgs/cudnn-10.0-v7.6.3.30/lib64:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-12.1

source ~/.bashrc
# source $(poetry env info --path)/bin/activate

export NCCL_IB_DISABLE=1  # disable InfiniBand (the Vector cluster does not have it)
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export NCCL_ASYNC_ERROR_HANDLING=1 # set to 1 for NCCL backend
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

export MASTER_ADDR=$(hostname --fqdn)
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"

export PYTHONPATH="."
nvidia-smi

python main.py