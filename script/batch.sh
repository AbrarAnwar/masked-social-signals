#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodelist=dill-sage
#SBATCH --time=4000
#SBATCH --output=/home/tangyimi/masked-social-signals/slurm_output/batch_%j.out

python -m preprocess.batch --window=18 --stride=9 --version=v4
python -m preprocess.batch --window=9 --stride=5 --version=v4
python -m preprocess.batch --window=6 --stride=3 --version=v4