#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodelist=lime-mint
#SBATCH --time=4000
#SBATCH --output=/home/tangyimi/masked-social-signals/slurm_output/vqvae/1024_%j.out
#SBATCH --job-name=vq1024

# python -m experiment.train --model=vqvae --sweep=vqvae_29 --test_idx=29 &
# python -m experiment.train --model=vqvae --sweep=vqvae_28 --test_idx=28 &

python -m experiment.train --model=vqvae --sweep=vqvae_1024
