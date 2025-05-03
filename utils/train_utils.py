# -----------------------------------------------------------
# CVAE Training Utilities
# -----------------------------------------------------------
# This script provides core helper functions for training a CVAE, including
#
# 1. cvae_loss_function:
#    - combines binary cross entopy and KL divergence to optimize reconstruction
#      quality while regularizing the latent space
#
# 2. save_model:
#    - saves the model's learned paramters to a checkpoint file for alter use or infernece
# 3. plot_losses:
#    - visualizes training loss over epochs to help monitor learning progress or detect overfitting
#
# These functions are designed to keep the main training script clean, modular, and easy to maintain


# --- Imports ---
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


def cvae_loss_function(recon_x, x, mu, logvar):
    '''
    Computes the CVAE loss:
    1. Reconstruction loss (Binary Cross Entropy) - how close the output is to the original image
    2. KL Divergence - how close the learned distribution is to a standard normal
    '''
    #flatten original and reconstructed images to 3D (batch_size, num_features)
    #this isnecessary because the loss function expects flat vectors, not image tensors (C, H, W)
    x = x.view(x.size(0), -1)
    recon_x = recon_x.view(recon_x.size(0), -1)

    #binary cross entropy: compares the original image to the reconstruction
    #measures how well each pizel was predicted -  ideal for normalized outputs in [0, 1]
    #'sum' reduction adds all pixel-wise errors into a single scalar loss
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')

    #KL divergence: regularizes the latent space to be close to standard normal (N(0, 1))
    #this helps ensure the latent vectors z are useful for smooth generation 
    #formula is dervied from the KL divergence between two Gaussians
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    #total loss
    return bce + kld



def save_model(model, path):
    '''
    Saves model weights to a file.
    Usefulf or checkpoints and reloading trained models later.
    '''
    #save only the model's learned paramters (weights and biases), not the entire model class
    #this is more flexible and lightweight
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")



def plot_losses(losses, save_path=None):
    '''
    Plots and optionally saves training loss over epochs
    '''
    #create a new figure for plotting
    plt.figure()
    #plot the loss values over time (one value per epoch)
    plt.plot(losses, label="Training Loss")

    #label the x-axis as "Epoch" to show training progress
    plt.xlabel("Epoch")
    
    #label the y-axis as "Loss" to show how the model is improving
    plt.ylabel("Loss")

    #add a title to the plot
    plt.title("CVAE Training Loss")

    #show the legend
    plt.legend()

    #if a save path is provided, create the directory (if it doesn't exist) and save the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    #otherwise, just display the plot on screen 
    else:
        plt.show()