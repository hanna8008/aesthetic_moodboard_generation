# -----------------------------------------------------------
# CVAE Training Script
# -----------------------------------------------------------
#
# Trains a CVAE using mood and/or color-labeled image data. Loads config
# settings, builds the model, runs training, and saves the model + logs
# to output folders

# --- Imports ---
import os
import sys
import yaml
import torch
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils.dataset import load_dataset
from utils.train_utils import cvae_loss_function, save_model, plot_losses
from model.cvae import CVAE 
import matplotlib.pyplot as plt
import numpy as np
from utils.dataset import get_condition_vector_dual

print("CVAE training script started.")

# --- Load Config ---
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set device: this checks for GPU and defaults to CPU if not available
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

# For reproducibility
torch.manual_seed(config["seed"])


from utils.dataset import load_dataset

# --- Load and Split Dataset ---
full_dataset = load_dataset(
    csv_path=config["csv_path"],
    image_root=config["image_root"],
    condition_type=config["condition_type"]
)



print(f"Loaded dataset length: {len(full_dataset)}")

# Split into 70% train, 15% val, 15% test
from torch.utils.data import random_split
train_len = int(0.7 * len(full_dataset))
val_len = int(0.15 * len(full_dataset))
test_len = len(full_dataset) - train_len - val_len
train_subset, val_subset, test_subset = random_split(full_dataset, [train_len, val_len, test_len])
print(f"Train/Val/Test subsets complete.")

# Wrap each subset in a DataLoader
train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
val_loader = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
test_loader = DataLoader(test_subset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

print(f"Dataset split: {len(train_subset)} train | {len(val_subset)} val | {len(test_subset)} test")
print("cond_dim from config:", config.get("cond_dim"))


# --- Initialize Model and Optimizer ---
model = CVAE(
    input_dim=config["input_dim"],
    condition_dim=config["cond_dim"],
    latent_dim=config["latent_dim"]
).to(device)

optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
losses = []


# --- Training Loop ---
print("Starting training...")

for epoch in range(config["num_epochs"]):
    model.train()
    running_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}"):
        x, c = batch
        x, c = x.to(device), c.to(device)

        optimizer.zero_grad()
        x_recon, mu, logvar = model(x, c)
        loss = cvae_loss_function(x_recon, x, mu, logvar)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader.dataset)
    losses.append(avg_loss)
    print(f"Epoch {epoch + 1}: Avg Training Loss = {avg_loss:.4f}")

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, c in val_loader:
            x, c = x.to(device), c.to(device)
            x_recon, mu, logvar = model(x, c)
            loss = cvae_loss_function(x_recon, x, mu, logvar)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader.dataset)
    print(f"Epoch {epoch + 1}: Avg Validation Loss = {avg_val_loss:.4f}")
    model.train()

    if (epoch + 1) % config["save_every"] == 0:
        os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)
        save_model(model, config["checkpoint_path"])

    # --- Save Progress Image ---
    if (epoch + 1) % 25 == 0:
        model.eval()
        z = torch.randn(1, config["latent_dim"]).to(device)

        test_mood = "dreamy"
        test_color = "blue"
        test_condition = get_condition_vector_dual(test_mood, test_color, config).unsqueeze(0).to(device)

        with torch.no_grad():
            gen = model.decode(z, test_condition)
            gen = torch.sigmoid(gen).view(3, 64, 64).cpu().numpy()

        # Construct unique filename using project name, epoch, mood, and color
        job_name = config.get("project_name", "unnamed")
        filename = f"progress_{job_name}_e{epoch+1}_{test_mood}_{test_color}.png"
        save_path = os.path.join(config["generated_dir"], filename)
        os.makedirs(config["generated_dir"], exist_ok=True)

        plt.imsave(save_path, np.transpose(gen, (1, 2, 0)))
        print(f"Progress image saved to {save_path}")



# --- Save the Final Trained Model ---
save_model(model, config["checkpoint_path"])

# --- Plot and Save Losses ---
plot_losses(losses, save_path=os.path.join(config["log_dir"], "training_loss.png"))

# --- Evaluate on Test Set ---
print("Evaluating on test set...")
model.eval()
test_loss = 0.0

with torch.no_grad():
    for x, c in test_loader:
        x, c = x.to(device), c.to(device)
        x_recon, mu, logvar = model(x, c)
        loss = cvae_loss_function(x_recon, x, mu, logvar)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader.dataset)
print(f"Final Test Loss = {avg_test_loss:.4f}")
