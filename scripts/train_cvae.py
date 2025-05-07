# -----------------------------------------------------------
# CVAE Training Script
# -----------------------------------------------------------
#
# Trains a Conditional Variational Autoencoder (CVAE) using mood and/or color-labled image data.
# It laods configuration settings, prepares the dataset, builds the model, trains it over multiple epochs,
# and periodically saves progress outputs and model checkpoints.



# --- Import standard libraries and deep learning tools ---
import os
import sys
import yaml
import torch
from torch import optim
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
import pandas as pd

# --- Ensure script can import files from parent directory ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# -- Import custom modules ---
from utils.data_utils import load_dataset
from utils.train_utils import cvae_loss_function, save_model, plot_losses
from model.cvae import CVAE 
import matplotlib.pyplot as plt
import numpy as np
from utils.data_utils import get_condition_vector_dual



#statement to print during running train_cvae.sh (CVAE training)
print("CVAE training script started.")



# --- Load Training Configuration from config.yaml file ---
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)



# --- Determine computation Device
# Set device: this checks for GPU and defaults to CPU if not available
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")



# --- Set random seed for reproducibility ---
#ensures that the training process is repetable across runs
torch.manual_seed(config["seed"])



# --- Load and Split Dataset ---
#load full dataset using the custom loader
full_dataset = load_dataset(
    #path to metadata CSV
    csv_path=config["csv_path"],

    #path to actual iamge folder
    image_root=config["image_root"],

    #whether only mood or both mood + color
    condition_type=config["condition_type"]
)

#statement to print during running train_cvae.sh (CVAE training)
print(f"Loaded dataset length: {len(full_dataset)}")



# --- Split into train (70%), Validation (15%), and Test (15%) ---
#calculate 70% of the total dataset for training
train_len = int(0.7 * len(full_dataset))
#assign 15% of the data for validation
val_len = int(0.15 * len(full_dataset))
#remaining 15% goes to the test set
test_len = len(full_dataset) - train_len - val_len

#randomly split the dataset into three parts
train_subset, val_subset, test_subset = random_split(full_dataset, [train_len, val_len, test_len])
print(f"Train/Val/Test subsets complete.")

#wrap the subsets into DataLoaders to efficiently load data in batches
#making training DataLoader
train_loader = DataLoader(
    #provide training subset
    train_subset, 
    #number of samples per batch (from config)
    batch_size=config["batch_size"], 
    #shuffle training data each epoch for better generalization
    shuffle=True, 
    #number of parallel data loading threads (from config)
    num_workers=config["num_workers"])

#making validation DataLoader
val_loader = DataLoader(
    #provide validation subset
    val_subset, 
    #use same batch size for consistency
    batch_size=config["batch_size"], 
    #no need to shuffle validation data
    shuffle=False, 
    #use the same number of workers
    num_workers=config["num_workers"])

#making test DataLoader
test_loader = DataLoader(
    #provide test subset
    test_subset, 
    #use same batch size
    batch_size=config["batch_size"], 
    #no shuffling for test set
    shuffle=False, 
    #consistent number of workers
    num_workers=config["num_workers"])

#log counts for each split
print(f"Dataset split: {len(train_subset)} train | {len(val_subset)} val | {len(test_subset)} test")

#print the dimensionality of the condition vector
print("cond_dim from config:", config.get("cond_dim"))



# --- Build the CVAE Model and Move it to the Selected Device ---
model = CVAE(
    #size of flattened image
    input_dim=config["input_dim"],

    #size of the conditon vector
    condition_dim=config["cond_dim"],

    #size of the latent space z
    latent_dim=config["latent_dim"]

#move the model to the selected device (GPU or CPU)
).to(device)

#Use Adam optimizer, with specified learning rate, to efficiently update model weights.
#combines momentum and scaling to handle sparse gradients and noisy loss landscapes
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

#create empty list to track average loss at each epoch
losses = []



# --- Training Loop ---
#log that training is starting
print("Starting training...")

#loop through total number of epochs
for epoch in range(config["num_epochs"]):
    #set model to training mode (enables dropout, batchnorm, etc.)
    model.train()
    #initialize loss tracker for current epoch
    running_loss = 0.0

    #loop through training batches with progress bar
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}"):
        #unpack batch: x = image, c = condition vector
        x, c = batch
        #move data to selected device
        x, c = x.to(device), c.to(device)

        #clear any gradients from previous iteration
        optimizer.zero_grad()
        #forward pass: reconstruct image, get latent parameters
        x_recon, mu, logvar = model(x, c)
        #calculate total loss (reconstruction + KL divergence)
        loss = cvae_loss_function(x_recon, x, mu, logvar)
        #backpropogate gradients
        loss.backward()
        #update model weights using optimizer
        optimizer.step()

        #add batch loss to running total
        running_loss += loss.item()

    #calculate average training loss per sample
    avg_loss = running_loss / len(train_loader.dataset)
    #store average loss in list
    losses.append(avg_loss)
    #log training loss for the epoch
    print(f"Epoch {epoch + 1}: Avg Training Loss = {avg_loss:.4f}")



    # --- Validation ---
    #switch to evaluation mode (disables dropout, etc.)
    model.eval()
    #initialize validation loss tracker
    val_loss = 0.0

    #disable gradient calculation during validation
    with torch.no_grad():
        #loop through validation batches
        for x, c in val_loader:
            #move data to device
            x, c = x.to(device), c.to(device)
            #forward pass (no gradient tracking)
            x_recon, mu, logvar = model(x, c)
            #compute validation loss
            loss = cvae_loss_function(x_recon, x, mu, logvar)
            #accumulate validation loss
            val_loss += loss.item()

    #compute average validation loss
    avg_val_loss = val_loss / len(val_loader.dataset)
    #print validation performance
    print(f"Epoch {epoch + 1}: Avg Validation Loss = {avg_val_loss:.4f}")
    #set model back to training mode 
    model.train()

    #save model every N epochs
    if (epoch + 1) % config["save_every"] == 0:
        #define checkpoint filename
        epoch_path = f"outputs/checkpoints/cvae_dual_epoch_{epoch+1}.pth"
        #save model weights to disk
        save_model(model, epoch_path)



    # --- Save Progress Image ---
    #every 25 epochs, generate a test image
    if (epoch + 1) % 25 == 0:
        #set model to eval mode
        model.eval()
        #sample random latent vector
        z = torch.randn(1, config["latent_dim"]).to(device)

        #select a mood condition
        test_mood = "dreamy"
        #select a color condition
        test_color = "blue"
        #generate condition vector
        test_condition = get_condition_vector_dual(test_mood, test_color, config).unsqueeze(0).to(device)

        #no gradients needed for generation
        with torch.no_grad():
            #decode the image
            gen = model.decode(z, test_condition)
            #apply sigmoid and reshape to image format
            gen = torch.sigmoid(gen).view(3, 64, 64).cpu().numpy()

        #get project name from config or use default
        job_name = config.get("project_name", "unnamed")
        #create a filename using epoch and condition
        filename = f"progress_{job_name}_e{epoch+1}_{test_mood}_{test_color}.png"
        #create full file path
        save_path = os.path.join(config["generated_dir"], filename)
        #create output directory if it doesn't exist
        os.makedirs(config["generated_dir"], exist_ok=True)

        #save image (HWC format)
        plt.imsave(save_path, np.transpose(gen, (1, 2, 0)))
        #log image save path
        print(f"Progress image saved to {save_path}")



# --- Save the Final Trained Model ---
#save final model to specified checkpoint path
save_model(model, config[f"checkpoint_path_{epoch}"])



# --- Save preview images for selected mood-color pairs ---
#set model to evaluation mode
model.eval()
#predefine condition pairs
test_combos = [("dreamy", "blue"), ("natural", "green"), ("romantic", "red")]

#loop through test condition pairs
for mood, color in test_combos:
    #sample random z
    z = torch.randn(1, config["latent_dim"]).to(device)
    #get condition
    condition = get_condition_vector_dual(mood, color, config).unsqueeze(0).to(device)

    #disable gradients
    with torch.no_grad():
        #generate image
        output = model.decode(z, condition)
        #reshape output
        img = torch.sigmoid(output).view(3, 64, 64).cpu().numpy()

    #define path for the image to save
    img_path = f"../outputs/generated/progress_cvae_moodboard_generator_e{epoch+1}_{mood}_{color}.png"
    #save image in correct format
    plt.imsave(img_path, np.transpose(img, (1, 2, 0)))
    #log save
    print(f"Saved: {img_path}")



# --- Plot and Save Losses ---
#save loss curve to disk
plot_losses(losses, save_path=os.path.join(config["log_dir"], "training_loss.png"))



# --- Evaluate on Test Set ---
#start test evaluation
print("Evaluating on test set...")
#ensure model is in eval mode
model.eval()
#track total test loss
test_loss = 0.0

#no gradients needed
with torch.no_grad():
    #loop through test data
    for x, c in test_loader:
        #move data to device
        x, c = x.to(device), c.to(device)
        #forward pass
        x_recon, mu, logvar = model(x, c)
        #compute loss
        loss = cvae_loss_function(x_recon, x, mu, logvar)
        #accumulate loss
        test_loss += loss.item()

#calculate average test loss
avg_test_loss = test_loss / len(test_loader.dataset)
#log test performance
print(f"Final Test Loss = {avg_test_loss:.4f}")
