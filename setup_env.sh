#!/bin/bash

echo "Setting up CVAE Moodboard Environment..."

# Fix: No spaces around "="
ENV_NAME="moodgen"

# 0. Load Conda (required on Quest)
module purge
module load anaconda3

# 1. Create Conda Environment (only if it doesn't already exist)
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "Creating Conda Environment: $ENV_NAME..."
    conda create -n $ENV_NAME python=3.9 -y
fi

# 2. Activate Environment
echo "Activating Environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 3. Install Requirements
echo "Installing Python Packages from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Success Message
echo "Environment '$ENV_NAME' setup complete!"
echo "To activate it later, run: conda activate $ENV_NAME"
