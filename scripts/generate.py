import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import yaml

#add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from model.cvae import CVAE
from utils.dataset import get_condition_vector_dual

# --- Parse Arguments ---
parser = argparse.ArgumentParser(description="Generate image from mood and color conditions using trained CVAE.")
parser.add_argument("--mood", type=str, required=True, help="Mood label (e.g. 'dreamy')")
parser.add_argument("--color", type=str, required=True, help="Color label (e.g. 'pastel')")
parser.add_argument("--save_dir", type=str, default="outputs/generated/", help="Directory to save generated image")
args = parser.parse_args()

# --- Load Config ---
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

# --- Load Model ---
model = CVAE(
    input_dim=config["input_dim"],
    condition_dim=config["cond_dim"],
    latent_dim=config["latent_dim"]
).to(device)

model.load_state_dict(torch.load(config["checkpoint_path"], map_location=device))
model.eval()

# --- Get Conditioning Vector ---
condition = get_condition_vector_dual(args.mood, args.color, config).to(device)

# --- Sample Random Latent Vector z ~ N(0, I) ---
z = torch.randn(1, config["latent_dim"]).to(device)

# --- Generate Image ---
with torch.no_grad():
    generated_flat = model.decode(z, condition)
    generated_flat = torch.sigmoid(generated_flat)

generated_image = generated_flat.view(3, 64, 64).cpu().numpy()
#CHW -> HWC
generated_image = np.transpose(generated_image, (1, 2, 0)) 

# --- Save Image ---
os.makedirs(args.save_dir, exist_ok=True)
filename = f"{args.mood}_{args.color}.png"
save_path = os.path.join(args.save_dir, filename)

plt.imsave(save_path, generated_image)
print(f"Image saved to {save_path}")