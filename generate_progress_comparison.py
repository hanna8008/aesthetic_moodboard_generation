# -------------------------------------------------------
# Generate Progress Comparison Image
# -------------------------------------------------------
# 
# This script generates a series of images using a trained
# CVAE model at different training epochs to visualize how image
# generation quality evovles over time for a specific mood-color
# condition.
#
# It loads multiple model checkpoints (e.g., every 25 epochs), 
# samples a random latent vector, decodes the image using specified
# mood and color condition, and saves the output image



# --- Imports ---
import os
import torch
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt

from model.cvae import CVAE
from utils.dataset import get_condition_vector_dual



# --- Argument Parser ---
#create a parser to allow user to input mood and color at runtine
parser = argparse.ArgumentParser(description="Generate progress images for a mood-color combo over training epochs.")

#required mood argument (e.g., "dreamy")
parser.add_argument("--mood", type=str, required=True, help="Mood label (e.g., 'romantic')")

#requierd color argument (e.g., "blue")
parser.add_argument("--color", type=str, required=True, help="Color label (e.g., 'red')")

#parse the user-provided arguments
args = parser.parse_args()



# --- Config Paths ---
#path to the config.yaml file
config_path = "configs/config.yaml"

#folder where CVAE model checkpoitns are saved
checkpoint_dir = "outputs/checkpoints"

#fodler where generated images will be saved
output_dir = "outputs/generated"



# --- Load Config ---
#load configuration settings for model and training
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

#choose GPU if available, otherwise fallback to CPU
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")



# --- Prepare Inputs ---
#extract mood and color from user input
mood = args.mood
color = args.color

#define list of epochs at which models were saved
#e.g., [25, 50, ..., 200]
epochs = list(range(25, 201, 25))  

#create the one-hot encoded mood+color condition vector and add batch dimension
condition_vector = get_condition_vector_dual(mood, color, config).unsqueeze(0).to(device)



# --- Loop Over Epochs ---
#iterate through each specified epoch to load that checkpoint
for epoch in epochs:
    #build the path to the checkpoint file for this epoch
    checkpoint_path = os.path.join(checkpoint_dir, f"cvae_dual_epoch_{epoch}.pth")

    #if the file doesn't exist, skip this epoch
    if not os.path.exists(checkpoint_path):
        print(f"Missing: {checkpoint_path}")
        continue

    #rebuild the model architecture (matches the one used during training)
    model = CVAE(
        input_dim=config["input_dim"],
        condition_dim=config["cond_dim"],
        latent_dim=config["latent_dim"]
    ).to(device)

    #load the weights from the checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    #switch to evaluation mode (disable dropout/batchnorm randomness)
    model.eval()



    # --- Generate an Save Image ---
    #disable gradient tracking (inference only)
    with torch.no_grad():
        #sample random latent vector z from N(0, I)
        z = torch.randn(1, config["latent_dim"]).to(device)
        #generate image using decoder
        output = model.decode(z, condition_vector)
        #apply sigmoid and reshape to image format
        image = torch.sigmoid(output).view(3, 64, 64).cpu().numpy()

    #create the output filename for this epoch
    save_path = os.path.join(output_dir, f"progress_e{epoch}_{mood}_{color}.png")

    #save the image in (H, W, C) format using matplotlib
    plt.imsave(save_path, np.transpose(image, (1, 2, 0)))

    #log successful save
    print(f"Saved: {save_path}")
