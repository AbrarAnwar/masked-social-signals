#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodelist=lime-mint
#SBATCH --time=4000
#SBATCH --output=/home/tangyimi/masked-social-signals/slurm_output/kmeans/kmeans_%j.out
#SBATCH --job-name=kmeans


python -m experiment.train --model=kmeans --sweep=kmeans
