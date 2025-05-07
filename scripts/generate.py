# -----------------------------------------------------------
# CVAE Image Generation Script
# -----------------------------------------------------------
#
# This script uses a trained Conditional Variational Autoencoder (CAVE) to
# generate a new image based on user-provided mood and/or color conditions.
#
# It loads the trained model and config settings, prepares the conditioning 
# vector, samples a random latent vector, decodes the output, and saves the
# resulting aesthetic image to the specified output folder.
#
# Supports;
# - single-label conditioning (for mood; future = color)
# - dual-label conditioning (mood + color)


# --- Imports ---
import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

# --- Setup: add the project root directory to sys.path (import path) ---
#this ensures I can import from other folders like 'model/' and 'utils/'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# --- Model and Utility Imports ---
#import the trained CVAE model class and import helper functions to create condition vectors
from model.cvae import CVAE
from utils.data_utils import one_hot_encode
from utils.data_utils import get_condition_vector_dual




# --- Parse Arguments ---
#this section sets up command-line arguments the user provides when running the script 
parser = argparse.ArgumentParser(description="Generate image from mood and color conditions using trained CVAE.")

#requierd argument: mood label as a string (e.g., "dreamy", "adventurous", etc.)
parser.add_argument("--mood", type=str, required=True, help="Mood label (e.g. 'dreamy')")
#required argument: color label as a string (e.g., "blue", "red", etc.)
parser.add_argument("--color", type=str, required=True, help="Color label (e.g. 'pastel')")
#optional argument: where to save the generated image (default folder provided)
parser.add_argument("--save_dir", type=str, default="outputs/generated/", help="Directory to save generated image")
#parse the arguments from the command line and store them in the 'args' object
args = parser.parse_args()



# --- Load Config  from config.yaml---
#this reads the configuration file with model and training settings
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

#choose device: use GPU if available, otherwise fallback to CPU
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

#store the list of mood class labels from config
mood_classes = config["mood_labels"]



# --- Load and Initialize the CVAE Model ---
#the model must match the architecture and settings used during training
model = CVAE(
    #total number of pixels in flattened image (e.g. 3*64*64)
    input_dim=config["input_dim"],

    #total number of values in the condition vector (e.g., mood + color)
    condition_dim=config["cond_dim"],

    #size of the latent space (e.g., 128)
    latent_dim=config["latent_dim"]

#move model to GPU or CPU based on earlier check
).to(device)

#load the trained weights from the checkpoitn file
#map_location ensures compatibility with device even if trained on different hardware
model.load_state_dict(torch.load(config["checkpoint_path"], map_location=device))

#switch the model to evaluation mode (important to disable dropout or batchnorm randomness)
model.eval()



# --- Prepare and Get Conditioning Vector (Mood + Color) ---
#depending on config, condition may be single-label or dual-label:
# 1. both mood and color ("dual")
# 2. just mood ("mood")
# 3. just color ("color")

#unsqueeze(0).to(device) - add a batch dimension so the final shape is (1, cond_dim) required by the model,
# then move the tensor to the same device as the model (GPU or CPU)

#check if the user selected a dual-conditioning setup in the config file
if config["condition_type"] == "dual":
    #if so, combine both mood and color one-hot vectors
    #example result: [0, 1, 0, 0, 0, 1] where the first half is mood, second half is color
    condition = get_condition_vector_dual(args.mood, args.color, config).unsqueeze(0).to(device)

#check if the user selected a single-conditioning setup in the config file for just mood
elif config["condition_type"] == "mood":
    #if conditioning only mood, load list of mood labels from config
    mood_classes = config["mood_labels"]
    #convert the input mood (e.g., "dreamy") to a one-hot vector using known classes
    condition = one_hot_encode(args.mood, mood_classes).unsqueeze(0).to(device)

'''
single-conditioning: color only - implementing this if time

#check if the user selected a single-conditioning setup in the config file for just mood
elif config["condition_type"] == "color":
    #if conditioning only on color, load list of color labels from config
    color_classes = config["color_labels"]
    #convert the input color (e.g., "blue") to a one-hot vector using known classes
    condition = one_hot_encode(args.color, color_classes).unsqueeze(0).to(device)

else:
    #if config["condition_type"] is set to something invalid, raise an error to catch it 
    raise ValueError(f"Unsupported condition type: {config['condition_type']}. Choose from ['mood', 'color', 'dual'].")'''



# --- Generate a Random Latent Vector z ~ N(0, I) ---
#Simulate a point z in the latent space by drawing from a standard normal distribution 
#this is like telling the model: "generate a new image from scratch"
z = torch.randn(1, config["latent_dim"]).to(device)



# --- Generate Image from Latent Vector and Condition ---
#run the model's decoder to generate an image without computing gradient (inferencemode)
with torch.no_grad():
    #decode the random z and condition into a flattened image
    generated_flat = model.decode(z, condition)

    #apply sigmoid to ensure pixel values are in range [0, 1] (needed in case model skips this)
    generated_flat = torch.sigmoid(generated_flat)



# -- Reshape Output to Image Format ---
#output is currently flat, so reshape to (channels, height, width)
generated_image = generated_flat.view(3, 64, 64).cpu().numpy()

#rearrange axes from (C, H, W) --> (H, W, C) for saving with matplotlib (HWC = image format)
generated_image = np.transpose(generated_image, (1, 2, 0)) 



# --- Save the Generated Image ---
#ensure the output directory exists and create it if it doesn't
os.makedirs(args.save_dir, exist_ok=True)

#build a descriptive filename based on the condition type
if config["condition_type"] == "dual":
    #include both mood and color in filename if using dual-conditioning
    filename = f"generated_{args.mood}_{args.color}.png"

elif config["condition_type"] == "mood":
    #include only mood if using single-conditioning
    filename = f"generated_{args.mood}.png"

#altering the image size before it is saved so that way the output of the image in the GUI is bigger
#convert NumPy array to PIL Image and resize it
#generated_image = Image.fromarray((generated_image).astype(np.uint8))
#generated_image = generated_image.resize((512, 512),  Image.NEAREST)

#join directory and filename into a full path
save_path = os.path.join(args.save_dir, filename)

#save the image using matplotlib - expects data in (H, W, C) format with values in [0, 1]
plt.imsave(save_path, generated_image)

#print confirmation so the user knows where to find the outupt
print(f"Image saved to {save_path}")