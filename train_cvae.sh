#!/bin/bash
#SBATCH --account=e32706
#SBATCH --job-name=train_cvae
#SBATCH --output=outputs/logs/train_%j.log
#SBATCH --time=1:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

echo "Starting .sh file"

#Activate conda or virtualenv here
module purge

#Use hook + absolute path
eval "$(/home/qlh2976/miniconda/bin/conda shell.bash hook)"
conda activate /home/qlh2976/miniconda/envs/moodgen

cd /home/qlh2976/spring2025/genAI/aesthetic_moodboard_generation

echo "Environment activated. Starting training..."

#Make sure current directory is treated as module root
export PYTHONPATH=$(pwd)

#Run training script
echo "Starting Python training script..."
python scripts/train_cvae.py

#End
echo "Training complete!"