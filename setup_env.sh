#!/bin/bash

echo "Setting up CVAE Moodboard Envrionment..."

#1. Create Virtual Envrionment
echo "Creating Virtual Envrionment..."
python3 -m venv venv

#2. Activate Envrionment
echo "Activating Envrionment..."
source venv/bin/activate

#3. Upgrade pip
pip install --upgrade pip

#4. Install Requirements
echo "Installing Python Packages from requirements.txt..."
pip install -r requirements.txt

#5. Success Message
echo "Envrionment setup complete!"
echo "To activate it later, run: source venv/bin/activate"