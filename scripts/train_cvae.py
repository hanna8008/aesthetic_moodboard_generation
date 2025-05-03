'''
Script to train the CVAE
#scripts/train_cvae.py

1. load imports and config
2. load data (from utils.dataset)
3. initialize model and optimizer
4. Training loop:
    - forward pass
    - compute loss
    - backward pass
    - save model periodically
    - track losses
5. save final model

use config in python by:

import yaml

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

batch_size = config['batch_size']
input_dim = config['input_dim']

'''

# -----------------------------------------------------------
# CVAE Training Script
# -----------------------------------------------------------
#
# Trains a CVAE using mood and/or color-labeled image data. Loads config
# settings, build the models, runstraining, and saves the model + logs
# to output folders

# --- Imports ---
import os
import yaml
import torch
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.dataset import load_dataset
from utis.train_utils import cvae_loss_function, save_model, plot_losses
from model.cvae import CVAE 



# --- Load Config ---
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

#set device 
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

#for reproducibility
torch.manual_seed(config["seed"])



# --- Load and Split Dataset ---
#load the list of (image_tensor, condition_tensor) pairs from the filtered dataset
#each sample will be used to train the CVAE with teh associated mood or color condition
full_dataset = load_dataset(
    csv_path = config["csv_path"],
    image_root = config["image_root"],
    condition_type = "mood"
)

#split into 70% train, 15% val, 15% tet
total_size = len(full_dataset)
train_end = int(0.7 * total_size)
val_end = train_end + int(0.15 * total_size)

train_subset = full_dataset[:train_end]
val_subset = full_dataset[train_end:val_end]
test_subset = full_dataset[val_end:]

#wrap each subset in a DataLoader
train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
val_loader = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
test_loader = DataLoader(test_subset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])


print(f"Dataset split: {len(train_subset)} train | {len(val_subset)} val | {len(test_subset)} test")



# --- Initialize Model and Optimizer ---
#create the CVAE model using dimensions from config
model = CVAE(
    #flattened image input (e.g., 3x64x64 = 12288)
    input_dim = config["input_dim"],
    #one-hot encoded label size
    condition_dim = config["cond_dim"],
    #size of latent space (z vector)
    latent_dim = config["latent_dim"]
#move model to GPU to CPU
).to(device)

#use Adam optimizer for efficient training
optimizer = optim.Adam(model.paramters(), lr=config["learning_rate"])

#track losses to visualize later
losses = []



# --- Trainign Loop ---
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

    #save checkpoing every few epochs
    if (epoch + 1) % config["save_every"] == 0:
        os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)
        save_model(model, config["checkpoint_path"])



# --- Save Final Model ---
save_model(model, config["checkpoint_path"])

# --- Plot and Save Losses ---
plot_losses(losses, save_path=os.path.join(config["log_dir"], "training_loss.png"))

# --- Evaluate on Test Set --
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