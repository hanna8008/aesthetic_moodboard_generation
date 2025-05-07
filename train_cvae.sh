#!/bin/bash
#SBATCH --account=e32706
#SBATCH --job-name=train_cvae
#SBATCH --output=outputs/logs/train_%j.log
#SBATCH --time=48:00:00
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=2

echo "Starting .sh file"

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Log file: outputs/logs/train_${SLURM_JOB_ID}.log"

#Activate conda or virtualenv here
module purge

#Use hook + absolute path
#eval "$(/home/qlh2976/miniconda/bin/conda shell.bash hook)"
#conda activate /home/qlh2976/miniconda/envs/moodgen"""
source $(conda info --base)/etc/profile.d/conda.sh
conda activate moodgen


#cd /home/qlh2976/spring2025/genAI/aesthetic_moodboard_generation
cd $SLURM_SUBMIT_DIR

echo "Environment activated. Starting training..."

#Make sure current directory is treated as module root
export PYTHONPATH=$(pwd)

#Run training script
echo "Starting Python training script..."
python scripts/train_cvae.py

#End
echo "Training complete!"