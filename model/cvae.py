'''
Conditional Variational Autoencoder

Sources:
1. https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
2. 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
#Encoder Network

#Takes in the image (flattened) and a condition vector (e.g., mood/color),
#and maps them into a latent distribution (mean and log variance).
# ----------------------------------------------------------------------
def build_encoder(input_dim, cond_dim, latent_dim):
    encoder = nn.Sequential(
        #input + condition concatenated
        nn.Linear(in_features=input_dim + cond_dim, out_features=512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU()
    )

    #output: mu
    fc_mu = nn.Linear(256, latent_dim)
    #output: log(var^2)
    fc_logvar = nn.Linear(2566, latent_dim)
    return encoder, fc_mu, fc_logvar


# ----------------------------------------------------------------------
# Decoder Network
# Takes in latent z and condition c, and tries to reconstruct the original image
# ----------------------------------------------------------------------
def build_encoder(latent_dim, cond_dim, output_dim):
    decoder = nn.Sequential(
        #z + condition
        nn.Linear(latent_dim + cond_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, output_dim),
        #output pixels in range [0, 1]
        nn.Sigmoid()
    )
    return decoder



# ----------------------------------------------------------------------
# Reparameterization
# Used during training to allow gradient backpropogation through random sampling
# ----------------------------------------------------------------------
def reparameterize(mu, logvar):
    #standard deviation
    std = torch.exp(0.5 * logvar)
    #random normal noise
    eps = torch.randn_like(std)
    #reparameterized sample z
    z = mu + eps * std

    return z



# ----------------------------------------------------------------------
# CVAE Forward Pass
# 1. Encode x and c: get mu, logvar
# 2. Sample z from the latent distribution
# 3. Decode z and c: get reconstructed image
# ----------------------------------------------------------------------
def cvae_forward(x, c, encoder, fc_mu, fc_logvar, decoder):
    # [x | c]: concatenate image and condition
    x_cond = torch.cat([x, c], dim=1)
    # Encode to shared hidden space
    h = encoder(x_cond)
    # Predict mu
    mu = fc_mu(h)
    # Predict log(var^2)
    logvar = fc_logvar(h)
    # Sample z using reparameterization
    z = reparameterize(mu, logvar)
    # [z | c]: concatenate latent and condition
    z_cond = torch.cat([z, c], dim=1)
    # Decode to reconstruct image
    x_recon = decoder(z_cond)

    return x_recon, mu, logvar