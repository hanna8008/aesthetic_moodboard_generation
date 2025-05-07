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
#
# The CVAE architecture for conditonal image generation includes:
# 1. An encoder: compress the input image + condition into a latent dtribution (mu and logvar)
# 2. A reparameterization step: samples from the latent distribution while keeping it differentiable
# 3. A decoder: reconstructs the input image from the sampled latent vector and condition
#
# The CVAE can be trained on any dataset where each image is associated with a one-hot 
# condition label (e.g., moood, emotino, color). This supports controlled generation based
# on the label input.



# --- Imports ---
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F



# --- Model Definition ---
class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim):
        '''
        Initializes the Conditional Variational Autoencoder

        Arguments:
            - input_dim (int): flattened size of theinput image (e..g, 64x64x3 = 12288)
            - condition_dim (int): size of one-hot encoded condition vector (e.g., )
            - latent _dim (int): size of the latent representation (z-vector)
        '''
        super(CVAE, self).__init__()

        #store input parameters
        self.cond_dim = condition_dim
        self.input_dim =input_dim

        #debugging aid: print architecture setup on init
        #print(f"CVAE initialized with input_dim={input_dim}, cond_dim={condition_dim}, latent_dim={latent_dim}")


        # --- Encoder ---
        #input: concatenation of (flattened image + condition vector)
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
        #input: concatenation of (latent vector z + condition vector)
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
        Combines input image and condition into a single tensor, passes it through the encoder to calculate latent mean 
        and variance.

        Arguments:
            - x (Tensor): input image batch, shape (B, C*H*W)
            - c (Tensor): one-hot condition vector, shape (B, cond_dim)

        Returns:
            - mu (Tensor): mean of latent distribution, shape (B, latent_dim)
            - logvar (Tensor): log-variance of latent distribution, shape (B, latent_dim)
        '''
        #flatten size: x to 2D
        x = x.view(x.size(0), -1)

        #join image and condition into one input
        x_cond = torch.cat([x, c], dim=1)

        #print("x_flat:", x.shape, "c:", c.shape, "→ x_cond:", x_cond.shape)
        #pass through encoder to get hidden features
        h = self.encoder(x_cond)

        #predict mean of z
        mu = self.mu(h)

        #predict log(variance) of z
        logvar = self.logvar(h)

        #output latent distribution parameters
        return mu, logvar
    


    def reparameterize(self, mu, logvar):
        '''
        Samples a latent variable z using the reparameterization: z = mu + std * eps, 
        so we can backprop through random sampling

        What reparameterization does:
        Tricks to sample a latent vector z. So instead of sampling z randomly (which would stop gradients from flowing),
        we rewrite the sampling as: z = mu + std * eps where eps ~ N(0, 1)

        This lets the model "pretend" it's sampling, but in a way that still allows learning

        Allows gradients to pass through random smapling during training.

        Arguments:
            - mu (Tensor): mean of the latent distribution
            - logvar (Tensor): log variance of the latent distribution

        Returns:
            - z (Tensor): sampled latent vector
        '''
        #convert log-variance to standard deviation
        std = torch.exp(0.5 * logvar)

        #sample noise from standard normal (same shape as std)
        eps = torch.rand_like(std)

        #sample from z using mean and std
        return mu + eps*std



    def decode(self, z, c):
        '''
        Combines latent vector z and condition, and reconstructs the image (input)

        Arguments:
            - z (Tensor): latent vector (B, latent_dim)
            - c (Tensor): one-hot condition vector (B, cond_dim)

        Returns:
            - recon_x (Tensor): reconstructed image (B, input_dim)
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

        Arguments:
            - x (Tensor): input image batch
            - c (Tensor): condition vector vatch

        Returns:
            - x_recon (Tensor): reconstructed image
            - mu (Tensor): mean of latent distribution
            - logvar (Tensor): log variance of latent distribution
        '''
        #step 1: get latent space parameters
        mu, logvar = self.encode(x, c)

        #step 2: sample from latent distribution
        z = self.reparameterize(mu, logvar)

        #step 3: reconstruct image - decode latent vector to reconstruct input
        x_recon = self.decode(z, c)

        #return output and latent values (used in loss) - aka the reconstructed image and latent variables for loss computation
        return x_recon, mu, logvar