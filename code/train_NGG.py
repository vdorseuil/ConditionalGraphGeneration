import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from model.autoencoder import VariationalAutoEncoder
from model.denoise_model import DenoiseNN, p_losses
from torch_geometric.loader import DataLoader
from utils.data_processing import preprocess_dataset
from utils.noise_schedules import linear_beta_schedule

###############################################################################

# Argument parser
parser = argparse.ArgumentParser(description='NeuralGraphGenerator')

# Argument parser for configuring the NeuralGraphGenerator model
parser = argparse.ArgumentParser(description='Configuration for the NeuralGraphGenerator model')

# Learning rate for the optimizer
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for the optimizer, typically a small float value (default: 0.001)")

# Dropout rate
parser.add_argument('--dropout', type=float, default=0.0, help="Dropout rate (fraction of nodes to drop) to prevent overfitting (default: 0.0)")

# Batch size for training
parser.add_argument('--batch-size', type=int, default=256, help="Batch size for training, controlling the number of samples per gradient update (default: 256)")

# Number of epochs for the autoencoder training
parser.add_argument('--epochs-autoencoder', type=int, default=200, help="Number of training epochs for the autoencoder (default: 200)")

# Hidden dimension size for the encoder network
parser.add_argument('--hidden-dim-encoder', type=int, default=256, help="Hidden dimension size for encoder layers (default: 64)")

# Hidden dimension size for the decoder network
parser.add_argument('--hidden-dim-decoder', type=int, default=256, help="Hidden dimension size for decoder layers (default: 256)")

# Dimensionality of the latent space
parser.add_argument('--latent-dim', type=int, default=2, help="Dimensionality of the latent space in the autoencoder (default: 32)")

# Maximum number of nodes of graphs
parser.add_argument('--n-max-nodes', type=int, default=50, help="Possible maximum number of nodes in graphs (default: 50)")

# Number of layers in the encoder network
parser.add_argument('--n-layers-encoder', type=int, default=3, help="Number of layers in the encoder network (default: 2)")

# Number of layers in the decoder network
parser.add_argument('--n-layers-decoder', type=int, default=3, help="Number of layers in the decoder network (default: 3)")

# Dimensionality of spectral embeddings for graph structure representation
parser.add_argument('--spectral-emb-dim', type=int, default=10, help="Dimensionality of spectral embeddings for representing graph structures (default: 10)")

# Number of training epochs for the denoising model
parser.add_argument('--epochs-denoise', type=int, default=100, help="Number of training epochs for the denoising model (default: 100)")

# Number of timesteps in the diffusion
parser.add_argument('--timesteps', type=int, default=500, help="Number of timesteps for the diffusion (default: 500)")

# Hidden dimension size for the denoising model
parser.add_argument('--hidden-dim-denoise', type=int, default=512, help="Hidden dimension size for denoising model layers (default: 512)")

# Number of layers in the denoising model
parser.add_argument('--n-layers_denoise', type=int, default=3, help="Number of layers in the denoising model (default: 3)")

# Dimensionality of conditioning vectors for conditional generation
parser.add_argument('--dim-condition', type=int, default=128, help="Dimensionality of conditioning vectors for conditional generation (default: 128)")

# Number of conditions used in conditional vector (number of properties)
parser.add_argument('--n-condition', type=int, default=7, help="Number of distinct condition properties used in conditional vector (default: 7)")

# Beta value for the VAE loss
parser.add_argument('--beta', type=float, default=0.05, help="Beta value for the VAE loss (default: 1.0)")

args = parser.parse_args()

###############################################################################



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# preprocess train data, validation data and test data. Only once for the first time that you run the code. Then the appropriate .pt files will be saved and loaded.
trainset = preprocess_dataset("train", args.n_max_nodes, args.spectral_emb_dim)
validset = preprocess_dataset("valid", args.n_max_nodes, args.spectral_emb_dim)
testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim)



# initialize data loaders
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)


# initialize VGAE model
autoencoder = VariationalAutoEncoder(args.spectral_emb_dim+1, args.hidden_dim_encoder, args.hidden_dim_decoder, args.latent_dim, args.n_layers_encoder, args.n_layers_decoder, args.n_max_nodes).to(device)


optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)


train_losses = []
train_recon_losses = []
train_kl_losses = []
val_losses = []
val_recon_losses = []
val_kl_losses = []

# Train VGAE model
best_val_loss = np.inf
for epoch in range(1, args.epochs_autoencoder+1):
    autoencoder.train()
    train_loss_all = 0
    train_count = 0
    train_loss_all_recon = 0
    train_loss_all_kld = 0
    cnt_train=0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss, recon, kld  = autoencoder.loss_function(data, beta = args.beta)
        train_loss_all_recon += recon.item()
        train_loss_all_kld += kld.item()
        cnt_train+=1
        loss.backward()
        train_loss_all += loss.item()
        train_count += torch.max(data.batch)+1
        optimizer.step()
        
    autoencoder.eval()
    val_loss_all = 0
    val_count = 0
    cnt_val = 0
    val_loss_all_recon = 0
    val_loss_all_kld = 0

    for data in val_loader:
        data = data.to(device)
        loss, recon, kld  = autoencoder.loss_function(data, beta = args.beta)
        val_loss_all_recon += recon.item()
        val_loss_all_kld += kld.item()
        val_loss_all += loss.item()
        cnt_val+=1
        val_count += torch.max(data.batch)+1

    if epoch % 1 == 0:
        dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print('{} Epoch: {:04d}, Train Loss: {:.5f}, Train Reconstruction Loss: {:.2f}, Train KLD Loss: {:.2f}, Val Loss: {:.5f}, Val Reconstruction Loss: {:.2f}, Val KLD Loss: {:.2f}'.format(dt_t,epoch, train_loss_all/cnt_train, train_loss_all_recon/cnt_train, train_loss_all_kld/cnt_train, val_loss_all/cnt_val, val_loss_all_recon/cnt_val, val_loss_all_kld/cnt_val))
        

    train_losses.append(train_loss_all/cnt_train)
    train_recon_losses.append(train_loss_all_recon/cnt_train)
    train_kl_losses.append(train_loss_all_kld/cnt_train)
    val_losses.append(val_loss_all/cnt_val)
    val_recon_losses.append(val_loss_all_recon/cnt_val)
    val_kl_losses.append(val_loss_all_kld/cnt_val)

    if best_val_loss >= val_loss_all:
        best_val_loss = val_loss_all
        if os.path.exists("models") == False:
            os.makedirs("models")
        torch.save({
            'state_dict': autoencoder.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, './models/autoencoder_smallbeta.pth.tar')


autoencoder.eval()



# define beta schedule
betas = linear_beta_schedule(timesteps=args.timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# initialize denoising model
denoise_model = DenoiseNN(input_dim=args.latent_dim, hidden_dim=args.hidden_dim_denoise, n_layers=args.n_layers_denoise, n_cond=args.n_condition, d_cond=args.dim_condition).to(device)
optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)

# Train denoising model
best_val_loss = np.inf
for epoch in range(1, args.epochs_denoise+1):
    denoise_model.train()
    train_loss_all = 0
    train_count = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        x_g = autoencoder.encode(data)
        t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
        loss = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
        loss.backward()
        train_loss_all += x_g.size(0) * loss.item()
        train_count += x_g.size(0)
        optimizer.step()

    denoise_model.eval()
    val_loss_all = 0
    val_count = 0
    for data in val_loader:
        data = data.to(device)
        x_g = autoencoder.encode(data)
        t = torch.randint(0, args.timesteps, (x_g.size(0),), device=device).long()
        loss = p_losses(denoise_model, x_g, t, data.stats, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")
        val_loss_all += x_g.size(0) * loss.item()
        val_count += x_g.size(0)

    if epoch % 5 == 0:
        dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(dt_t, epoch, train_loss_all/train_count, val_loss_all/val_count))

    if best_val_loss >= val_loss_all:
        best_val_loss = val_loss_all
        torch.save({
            'state_dict': denoise_model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, './models/denoise_model_smallbeta.pth.tar')


if os.path.exists("plots") == False:
    os.makedirs("plots")

# Plot the training and validation losses
plt.figure(figsize=(10, 5))
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(train_losses, label='Training loss', color='blue')
plt.plot(val_losses, label='Validation loss', color = 'red')
plt.legend()
plt.savefig('plots/NGG_losses_smallbeta.png')


plt.figure(figsize=(10, 5))
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(train_recon_losses, label='Training reconstruction loss', color = 'blue')
plt.plot(val_recon_losses, label='Validation reconstruction loss', color = 'red')
plt.legend()
plt.savefig('plots/NGG_recon_losses_smallbeta.png')


plt.figure(figsize=(10, 5))
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# plt.ylim([0.0, 1.0])
plt.plot(train_kl_losses, label='Training KL loss', color = 'blue')
plt.plot(val_kl_losses, label='Validation KL loss', color = 'red')
plt.legend()
plt.savefig('plots/NGG_kl_losses_smallbeta.png')