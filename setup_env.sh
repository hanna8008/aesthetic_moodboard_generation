#!/bin/bash

# -----------------------------------------------------------
# Setup Script for CVAE Conda Environment
# -----------------------------------------------------------
#
# This script sets up a Python 3.9 Conda envrionment named "moodgen"
# and installs all required Python packages listed in requirements.txt.
# Designed for use on Quest or any system with Anaconda installed.



echo "Setting up CVAE Moodboard Environment..."


# --- Set Envrionment Name
ENV_NAME="moodgen"



# --- Step 0: Load Conda Module ---
#ensures the 'conda' command is available in this shell session
module purge
module load anaconda3



# --- Step 1: Create Conda Envrionment (if it doesn't already exist) ---
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "Creating Conda Environment: $ENV_NAME..."
    conda create -n $ENV_NAME python=3.9 -y
fi



# --- Step 2: Activate the Envrionment ---
#this allows pip and Python commands to apply to 'moodgen'
echo "Activating Environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME



# --- Step 3: Install Required Python Packages ---
#upgrade pip and install all dependencies from requirements.txt
echo "Installing Python Packages from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt



# --- Step 4: Confirmation Message ---
echo "Environment '$ENV_NAME' setup complete!"
echo "To activate it later, run: conda activate $ENV_NAME"
