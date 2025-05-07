#!/bin/bash

#Activate envrionment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate moodgen

#Run the GUI
python gui.py