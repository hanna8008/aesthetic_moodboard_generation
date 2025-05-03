#!/bin/bash
#SBATCH --account=p32562
#SBATCH --job-name=train_cvae
#SBATCH --output=outputs/logs/train_%j.log
#SBATCH --time=1:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

#Activate conda or virtualenv here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cvae_moodboard_env

#Run training script
python scripts/train_cvae.py