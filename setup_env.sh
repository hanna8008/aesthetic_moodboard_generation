#!/bin/bash

echo "Setting up CVAE Moodboard Envrionment..."

ENV_NAME = "moodgen"

#1. Create Virtual Envrionment
echo "Creating Conda Envrionment: $ENV_NAME..."
conda create -n $ENV_NAME python=3.9 -y

#2. Activate Envrionment
echo "Activating Envrionment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

#3. Install Requirements
echo "Installing Python Packages from requirements.txt..."
pip install -r requirements.txt

#4. Success Message
echo "Envrionment '$ENV_NAME' setup complete!"
echo "To activate it later, run: conda activate moodgen"