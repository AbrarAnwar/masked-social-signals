#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --gres=gpu:a6000:4
#SBATCH --nodelist=dill-sage
#SBATCH --time=4200
#SBATCH --output=/home/tangyimi/masked-social-signals/slurm_output/ablation/2_90_%j.out
#SBATCH --job-name=2_90

find_free_port() {
    python -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
}

# Find a free port
PORT1=$(find_free_port)

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3


torchrun --nproc_per_node=4 --master_port=$PORT1 -m experiment.train --model=masktransformer --sweep=2_90 --batch_path=dining_dataset/batch_window6_stride3_v4 --feature_mask=pose_only
# torchrun --nproc_per_node=4 --master_port=$PORT1 -m experiment.train --model=masktransformer --sweep=4_270 -ab