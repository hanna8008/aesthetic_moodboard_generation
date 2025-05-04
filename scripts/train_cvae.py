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
import pandas as pd

from utils.dataset import load_dataset
from utils.train_utils import cvae_loss_function, save_model, plot_losses
from model.cvae import CVAE 


print("CVAE training script started.")


# --- Load Config ---
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

#set device: this checks for GPU and defaults to CPU if not available, which is crucial for performance 
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

#print(f"Loaded dataset length: {len(load_dataset)}")

#df = pd.read_csv(config["csv_path"])

#split into 70% train, 15% val, 15% tet
total_size = len(full_dataset)
train_end = int(0.7 * total_size)
val_end = train_end + int(0.15 * total_size)
print(f"Splitting of data complete.")

train_subset = full_dataset[:train_end]
val_subset = full_dataset[train_end:val_end]
test_subset = full_dataset[val_end:]
print(f"Train/Val/Test subsets complete.")

#wrap each subset in a DataLoader
#shuffling helps the model generalize by prevening it from seing data in the same order every epoch
train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
val_loader = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
test_loader = DataLoader(test_subset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])


print(f"Dataset split: {len(train_subset)} train | {len(val_subset)} val | {len(test_subset)} test")

print("cond_dim from config:", config.get("cond_dim"))  # Should print 16


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
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

#track losses to visualize later
losses = []



# --- Trainign Loop ---
print("Starting training...")

#repeating training for the number of epochs defined in config
for epoch in range(config["num_epochs"]):
    #set the model to training mode (enables dropout, batchnorm updates, etc.)
    model.train()
    #initialize cumulative loss tracker for this epoch
    running_loss = 0.0

    #loop over batches of training data
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}"):
        #unpack image tensors and condition vectors
        x, c = batch
        #move data to GPU/CPU depending on the config
        x, c = x.to(device), c.to(device)

        #clear old gradients from previous batch
        optimizer.zero_grad()
        #forward pass: reconstruct x given x and condition c
        x_recon, mu, logvar = model(x, c)
        #compute total loss (reconstruction + KL Divergence)
        loss = cvae_loss_function(x_recon, x, mu, logvar)
        #backpropogate gradients throughteh network
        loss.backward()
        #update model weights using calculated gradients 
        optimizer.step()

        #accumulate the loss to calculate average later
        running_loss += loss.item()

    #compute average training per image
    avg_loss = running_loss / len(train_loader.dataset)
    #save the loss to plot the learning curve later
    losses.append(avg_loss)
    #output training loss for this epoch
    print(f"Epoch {epoch + 1}: Avg Training Loss = {avg_loss:.4f}")

    # --- Validation ---
    #set model to evaluation mode (turns of dropout, uses running stats in batchnorm)
    model.eval()
    #initialize cmulative validation loss
    val_loss = 0.0
    #disable gradient computation (saves memory and speeds up validation)
    with torch.no_grad():
        #loop over validation batches
        for x, c in val_loader:
            #move validation data to correct device
            x, c = x.to(device), c.to(device)
            #run model forward pass on validation data
            x_recon, mu, logvar = model(x, c)
            #compute validation loss
            loss = cvae_loss_function(x_recon, x, mu, logvar)
            #accumulate total validation loss
            val_loss += loss.item()
    #average validation loss per image
    avg_val_loss = val_loss / len(val_loader.dataset)
    #output validation performance
    print(f"Epoch {epoch + 1}: Avg Validation Loss = {avg_val_loss:.4f}")
    #switch back to training mode for next epoch
    model.train()

    #save checkpoing every few epochs (basedon config setting)
    if (epoch + 1) % config["save_every"] == 0:
        #ensure folder exists
        os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)
        #save model weights
        save_model(model, config["checkpoint_path"])



# --- Save the Final Trained Model ---
save_model(model, config["checkpoint_path"])

# --- Plot and Save Losses ---
#visualize training loss over epochs
plot_losses(losses, save_path=os.path.join(config["log_dir"], "training_loss.png"))

# --- Evaluate on Test Set --
print("Evaluating on test set...")
#set model to evaluation mode
model.eval()
#initialize test loss accumulator
test_loss = 0.0

#no need to compuate gradients during testing
with torch.no_grad():
    #loop through batches in test set
    for x, c in test_loader:
        #move test data to device
        x, c = x.to(device), c.to(device)
        #run model forward pass
        x_recon, mu, logvar = model(x, c)
        #compute test loss
        loss = cvae_loss_function(x_recon, x, mu, logvar)
        #add to total test loss
        test_loss += loss.item()

#average test loss per image
avg_test_loss = test_loss / len(test_loader.dataset)
#report final performance of the trained mode.
print(f"Final Test Loss = {avg_test_loss:.4f}")