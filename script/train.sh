#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodelist=dill-sage
#SBATCH --time=4000
#SBATCH --output=/home/tangyimi/masked-social-signals/slurm_output/vqvae_%j.out

#torchrun --nproc_per_node=8 -m experiment.train --model=vqvae --sweep_name=vqvae_v1
python -m experiment.train --model=vqvae --sweep_name=vqvae_42