#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --gres=gpu:a6000:8
#SBATCH --nodelist=dill-sage
#SBATCH --time=4000
#SBATCH --output=/home/tangyimi/masked-social-signals/slurm_output/multi_%j.out

source ~/anaconda3/etc/profile.d/conda.sh
conda activate social2

torchrun --nproc_per_node=8 -m experiment.main --feature_mask=multi --sweep_name=multi2