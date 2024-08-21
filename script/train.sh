#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --gres=gpu:a6000:4
#SBATCH --nodelist=dill-sage
#SBATCH --time=4000
#SBATCH --output=/home/tangyimi/masked-social-signals/slurm_output/train/kfold/mask_bite_%j.out

find_free_port() {
    python -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
}

# Find a free port
FREE_PORT=$(find_free_port)

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

# torchrun --nproc_per_node=4 --master_port=$FREE_PORT -m experiment.train --model=masktransformer --sweep=mask_gaze --feature_mask=mask_gaze
# torchrun --nproc_per_node=4 --master_port=$FREE_PORT -m experiment.train --model=masktransformer --sweep=mask_headpose --feature_mask=mask_headpose
# torchrun --nproc_per_node=4 --master_port=$FREE_PORT -m experiment.train --model=masktransformer --sweep=mask_pose --feature_mask=mask_pose
# torchrun --nproc_per_node=4 --master_port=$FREE_PORT -m experiment.train --model=masktransformer --sweep=mask_word --feature_mask=mask_word
# torchrun --nproc_per_node=4 --master_port=$FREE_PORT -m experiment.train --model=masktransformer --sweep=mask_speaker --feature_mask=mask_speaker
torchrun --nproc_per_node=4 --master_port=$FREE_PORT -m experiment.train --model=masktransformer --sweep=mask_bite --feature_mask=mask_bite


# torchrun --nproc_per_node=4 --master_port=$FREE_PORT -m experiment.train --model=masktransformer --sweep=kfmulti --feature_mask=multi
# torchrun --nproc_per_node=4 --master_port=$FREE_PORT -m experiment.train --model=masktransformer --sweep=gaze_only --feature_mask=gaze_only
# torchrun --nproc_per_node=4 --master_port=$FREE_PORT -m experiment.train --model=masktransformer --sweep=headpose_only --feature_mask=headpose_only
# torchrun --nproc_per_node=4 --master_port=$FREE_PORT -m experiment.train --model=masktransformer --sweep=pose_only --feature_mask=pose_only
# torchrun --nproc_per_node=4 --master_port=$FREE_PORT -m experiment.train --model=masktransformer --sweep=speaker_only --feature_mask=speaker_only 
# torchrun --nproc_per_node=4 --master_port=$FREE_PORT -m experiment.train --model=masktransformer --sweep=bite_only --feature_mask=bite_only

#python -m experiment.train --model=vqvae --sweep=vqvae_1d