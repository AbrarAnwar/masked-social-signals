#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodelist=glamor-ruby
#SBATCH --time=4000
#SBATCH --output=/home/tangyimi/masked-social-signals/slurm_output/gaze_autoencoder%j.out

# activate conda env
# conda init
source ~/anaconda3/etc/profile.d/conda.sh
conda activate social

# run script from above
python -m torch.distributed.run --nproc_per_node=1 /home/tangyimi/masked-social-signals/models/autoencoder.py --n_devices=1