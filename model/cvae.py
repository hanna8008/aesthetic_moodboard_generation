# -----------------------------------------------------------
# Conditional Variational Autoencdoer (CVAE) - script
# -----------------------------------------------------------
# This model learns to generate images based on a condition (like mood or color).
# It works by encoding both the image and its condition into a shared latent space,
# sampling a latent vector using reparameterization, and decoding it (with the same condition)
# to reconstruct the input image.
#
# The model is trained using a combination of reconstruction loss (how close the 
# output is to the input) and KL divergence (how close the learned distribution is
# to a normal distribution). This encourages smooth, structured latent space that
# supports controlled generation based on input conditions

# --- Imports ---
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim):
        super(CVAE, self).__init__()

        self.cond_dim = condition_dim
        self.input_dim =input_dim
        print(f"CVAE initialized with input_dim={input_dim}, cond_dim={condition_dim}, latent_dim={latent_dim}")

        # --- Encoder ---
        #input = image + condition vector
        #project it down to a smaller hidden representation
        self.encoder = nn.Sequential(
            #combine image and condition as input
            nn.Linear(self.input_dim + self.cond_dim, 512),
            #add non-linearity to capture complex patterns
            nn.ReLU(),
            #further compress features
            nn.Linear(512, 256),
            #activation again for depth
            nn.ReLU()
        )

        #latent mean and log-variance: two layers that generate parameters for the latent distribution
        #predicts the mean of z
        self.mu = nn.Linear(256, latent_dim)
        #predicts log(variance) of z (used to get std)
        self.logvar = nn.Linear(256, latent_dim)

        # --- Decoder ---
        #takes latent vector z (with condition info) and reconstructs the input image
        self.decoder = nn.Sequential(
            #combine z with condition
            nn.Linear(latent_dim + self.cond_dim, 256),
            nn.ReLU(),
            #expand to match original input size
            nn.Linear(256, 512),
            nn.ReLU(),
            #final output layer
            nn.Linear(512, input_dim),
            #scale output to range [0, 1] (image-like)
            nn.Sigmoid()
        )
    


    def encode(self, x, c):
        '''
        Combines input image and condition into a single tensor,
        passes it through the encoder to get latent mean and variance
        '''
        #flatten size: x to 2D
        x = x.view(x.size(0), -1)
        #join image and condition into one input
        x_cond = torch.cat([x, c], dim=1)
        #pass through encoder to get hidden features
        h = self.encoder(x_cond)
        #predict mean of z
        mu = self.mu(h)
        #predict log(variance) of z
        logvar = self.logvar(h)
        return mu, logvar
    


    def reparameterize(self, mu, logvar):
        '''
        Samples a latent variable z using the reparameterization:
        z = mu + std * eps, so we can backprop through random sampling
        '''
        #convert log-variance to standard deviation
        std = torch.exp(0.5 * logvar)
        #random noise with same shape as std
        eps = torch.rand_like(std)
        #sample from z using mean and std
        return mu + eps*std



    def decode(self, z, c):
        '''
        Combines latent vector z and condition, and reconstructs the image
        '''
        #join latent vector with condition
        z_cond = torch.cat([z, c], dim=1)
        #pass through decoder to get reconstructed image
        return self.decoder(z_cond)
        


    def forward(self, x, c):
        '''
        Run full CVAE pass:
        1. encode image + condition to get mu and logvar
        3. reparameterize to get latent vector z
        3. decode z + condition to get reconstructed image 
        '''
        #step 1: get latent space parameters
        mu, logvar = self.encode(x, c)
        #step 2: sample z
        z = self.reparameterize(mu, logvar)
        #step 3: reconstruct image
        x_recon = self.decode(z, c)
        #return output and latent values (used in loss)
        return x_recon, mu, logvar