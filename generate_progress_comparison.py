import os
import torch
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt

from model.cvae import CVAE
from utils.dataset import get_condition_vector_dual

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Generate progress images for a mood-color combo over training epochs.")
parser.add_argument("--mood", type=str, required=True, help="Mood label (e.g., 'romantic')")
parser.add_argument("--color", type=str, required=True, help="Color label (e.g., 'red')")
args = parser.parse_args()

# --- Config Paths ---
config_path = "configs/config.yaml"
checkpoint_dir = "outputs/checkpoints"
output_dir = "outputs/generated"

# --- Load Config ---
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

# --- Prepare Inputs ---
mood = args.mood
color = args.color
epochs = list(range(25, 201, 25))  # e.g., [25, 50, ..., 200]

condition_vector = get_condition_vector_dual(mood, color, config).unsqueeze(0).to(device)

# --- Loop Over Epochs ---
for epoch in epochs:
    checkpoint_path = os.path.join(checkpoint_dir, f"cvae_dual_epoch_{epoch}.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Missing: {checkpoint_path}")
        continue

    model = CVAE(
        input_dim=config["input_dim"],
        condition_dim=config["cond_dim"],
        latent_dim=config["latent_dim"]
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    with torch.no_grad():
        z = torch.randn(1, config["latent_dim"]).to(device)
        output = model.decode(z, condition_vector)
        image = torch.sigmoid(output).view(3, 64, 64).cpu().numpy()

    save_path = os.path.join(output_dir, f"progress_e{epoch}_{mood}_{color}.png")
    plt.imsave(save_path, np.transpose(image, (1, 2, 0)))
    print(f"Saved: {save_path}")
