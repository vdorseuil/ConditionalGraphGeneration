import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from model.contrastive_cvae import ContrastiveCVGAE
from torch_geometric.loader import DataLoader
from utils.data_processing import get_stats_mean_std, preprocess_dataset

###############################################################################

# Argument parser
parser = argparse.ArgumentParser(description='NeuralGraphGenerator')

# Argument parser for configuring the NeuralGraphGenerator model
parser = argparse.ArgumentParser(description='Configuration for the NeuralGraphGenerator model')

# Learning rate for the optimizer
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for the optimizer, typically a small float value (default: 0.001)")

# Maximum number of epochs for training
parser.add_argument('--max-epochs', type=int, default=200, help="Maximum number of epochs for training the model (default: 200)")

# Batch size for training
parser.add_argument('--batch-size', type=int, default=256, help="Batch size for training, controlling the number of samples per gradient update (default: 256)")

# Number of epochs for the autoencoder training
parser.add_argument('--epochs-autoencoder', type=int, default=200, help="Number of training epochs for the autoencoder (default: 200)")

# Hidden dimension size for the encoder network
parser.add_argument('--hidden-dim-stats-encoder', type=int, default=32, help="Hidden dimension size for encoder layers (default: 256)")

# Hidden dimension size for the statistics encoder network (contrastive learning)
parser.add_argument('--hidden-dim-encoder', type=int, default=256, help="Hidden dimension size for statistics encoder layers (default: 32)")

# Hidden dimension size for the decoder network
parser.add_argument('--hidden-dim-decoder', type=int, default=256, help="Hidden dimension size for decoder layers (default: 256)")

# Dimensionality of the latent space
parser.add_argument('--latent-dim', type=int, default=2, help="Dimensionality of the latent space in the autoencoder (default: 32)")

# Maximum number of nodes of graphs
parser.add_argument('--n-max-nodes', type=int, default=50, help="Possible maximum number of nodes in graphs (default: 50)")

# Number of layers in the encoder network
parser.add_argument('--n-layers-encoder', type=int, default=3, help="Number of layers in the encoder network (default: 3)")

# Number of layers in the stats encoder network (contrastive learning)
parser.add_argument('--n-layers-stats-encoder', type=int, default=3, help="Number of layers in the statistics encoder network (default: 3)")

# Number of layers in the decoder network
parser.add_argument('--n-layers-decoder', type=int, default=3, help="Number of layers in the decoder network (default: 3)")

# Dimensionality of spectral embeddings for graph structure representation
parser.add_argument('--spectral-emb-dim', type=int, default=10, help="Dimensionality of spectral embeddings for representing graph structures (default: 10)")

# Number of conditions used in conditional vector (number of properties)
parser.add_argument('--n-condition', type=int, default=7, help="Number of distinct condition properties used in conditional vector (default: 7)")

# Beta value for the VAE loss
parser.add_argument('--beta', type=float, default=0.05, help="Beta value for the VAE loss (default: 0.05)")

# Gamma value for the contrastive loss
parser.add_argument('--gamma', type=float, default=0.01, help="Beta value for the contrastive loss (default: 0.01)")

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

stats_mean, stats_std = get_stats_mean_std(trainset)

contrastive_cvgae = ContrastiveCVGAE(
    input_dim = args.spectral_emb_dim+1,
    hidden_dim_enc = args.hidden_dim_encoder,
    hidden_dim_dec = args.hidden_dim_decoder,
    hidden_dim_stats_enc = args.hidden_dim_stats_encoder,
    latent_dim = args.latent_dim,
    n_layers_enc = args.n_layers_encoder,
    n_layers_stats_enc = args.n_layers_stats_encoder,
    n_layers_dec = args.n_layers_decoder,
    n_max_nodes = args.n_max_nodes,
    n_cond = args.n_condition,
    stats_mean = stats_mean,
    stats_std = stats_std,
).to(device)

optimizer = torch.optim.Adam(contrastive_cvgae.parameters(), lr=args.lr)


train_losses = []
train_recon_losses = []
train_kl_losses = []
train_contrastive_losses = []
valid_losses = []
valid_recon_losses = []
valid_kl_losses = []
valid_contrastive_losses = []

best_val_loss = np.inf

for epoch in range(args.max_epochs):
    
    contrastive_cvgae.train()
    train_loss = 0
    train_recon_loss = 0
    train_kl_loss = 0
    train_contrastive_loss = 0
    beta = .05

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss, recon_loss, kl_loss, contrastive_loss = contrastive_cvgae.loss_function(data, beta = args.beta, gamma=args.gamma)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_recon_loss += recon_loss.item()
        train_kl_loss += kl_loss.item()
        train_contrastive_loss += contrastive_loss.item()

    train_loss /= len(train_loader)
    train_recon_loss /= len(train_loader)
    train_kl_loss /= len(train_loader)
    train_contrastive_loss /= len(train_loader)

    contrastive_cvgae.eval()
    valid_loss = 0
    valid_recon_loss = 0
    valid_kl_loss = 0
    valid_contrastive_loss = 0
    for data in val_loader:
        data = data.to(device)
        loss, recon_loss, kl_loss, contastive_loss = contrastive_cvgae.loss_function(data)
        valid_loss += loss.item()
        valid_recon_loss += recon_loss.item()
        valid_kl_loss += kl_loss.item()
        valid_contrastive_loss += contrastive_loss.item()

    valid_loss /= len(val_loader)
    valid_recon_loss /= len(val_loader)
    valid_kl_loss /= len(val_loader)
    valid_contrastive_loss /= len(val_loader)

    print(f'Epoch {epoch+1}/{args.max_epochs}, Train Loss: {train_loss:.4f}, Train Recon Loss: {train_recon_loss:.4f}, Train KL Loss: {train_kl_loss:.4f}, Train Contrastive Loss: {train_contrastive_loss:.4f} Valid Loss: {valid_loss:.4f}, Valid Recon Loss: {valid_recon_loss:.4f}, Valid KL Loss: {valid_kl_loss:.4f}, Valid Contrastive Loss: {valid_contrastive_loss:.4f}')

    train_losses.append(train_loss)
    train_recon_losses.append(train_recon_loss)
    train_kl_losses.append(train_kl_loss)
    train_contrastive_losses.append(train_contrastive_loss)
    valid_losses.append(valid_loss)
    valid_recon_losses.append(valid_recon_loss)
    valid_kl_losses.append(valid_kl_loss)
    valid_contrastive_losses.append(valid_contrastive_loss)

    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        if os.path.exists("models") == False:
            os.makedirs("models")
        torch.save({
            'state_dict': contrastive_cvgae.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, 'models/contrastive_cvgae.pth.tar')

if os.path.exists("plots") == False:
    os.makedirs("plots")

# Plot the training and validation losses
plt.figure(figsize=(10, 5))
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(train_losses, label='Training loss', color='blue')
plt.plot(valid_losses, label='Validation loss', color = 'red')
plt.legend()
plt.savefig('plots/contrastive_CVGAE_losses.png')


plt.figure(figsize=(10, 5))
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(train_recon_losses, label='Training reconstruction loss', color = 'blue')
plt.plot(valid_recon_losses, label='Validation reconstruction loss', color = 'red')
plt.legend()
plt.savefig('plots/contrastive-CVGAE_recon_losses.png')


plt.figure(figsize=(10, 5))
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(train_kl_losses, label='Training KL loss', color = 'blue')
plt.plot(valid_kl_losses, label='Validation KL loss', color = 'red')
plt.legend()
plt.savefig('plots/contrastive_CVGAE_kl_losses.png')

