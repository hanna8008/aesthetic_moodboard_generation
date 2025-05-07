#!/bin/bash

# -----------------------------------------------------------
# SLURM Submission Script for CVAE Training
# -----------------------------------------------------------
#
# This script is intended to run a CVAE training job on Quest's GPU
# nodes using the A100 GPU. It sets up the environment, activates the 
# Conda envrionment, and executes the training script.


#project account for resource tracking
#SBATCH --account=e32706
#job name displayed in SLURM queue
#SBATCH --job-name=train_cvae
#log file output (%j = job ID)
#SBATCH --output=outputs/logs/train_%j.log
#max wall time for the job
#SBATCH --time=48:00:00
#GPU partition to use
#SBATCH --partition=gengpu
#reuqest 1 A100 GPU
#SBATCH --gres=gpu:a100:1
#total memory required
#SBATCH --mem=40G
#number of CPU threads
#SBATCH --cpus-per-task=2



# --- Status Messagess ---
echo "Starting .sh file"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Log file: outputs/logs/train_${SLURM_JOB_ID}.log"



# --- Load Conda Envrionment ---
#unload existing modules to avoid conflicts
module purge

#activate the desired Conda envrionment for training 
source $(conda info --base)/etc/profile.d/conda.sh
conda activate moodgen



# --- Set working directory to job submission location ---
cd $SLURM_SUBMIT_DIR



# --- Set PYTHONPATH so that local modules can be imported ---
export PYTHONPATH=$(pwd)



echo "Environment activated. Starting training..."



# --- Launch the Training Script ---
echo "Starting Python training script..."
python scripts/train_cvae.py



# --- Completion Message ---
echo "Training complete!"