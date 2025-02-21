import argparse
import os

import torch
from model.autoencoder import VariationalAutoEncoder
from model.contrastive_cvae import ContrastiveCVGAE
from model.cvae import CVGAE
from utils.data_processing import get_stats_mean_std, preprocess_dataset
from utils.visuals import plot_latent

parser = argparse.ArgumentParser(description='LatentPlots')

parser.add_argument('--model', type=str, default='models/cvgae.pth.tar', help="Path of autoencoder model to load")
parser.add_argument('--model-type', type=str, default='cvgae', help="Type of model to load (cvgae, ngg or contrastive)")
parser.add_argument('--split', type=str, default='train', help="Dataset split to visualize (train, val, or test)")
parser.add_argument('--cond-idx', type=int, default=0, help="Index of condition property to visualize (default: 0)")

parser.add_argument('--batch-size', type=int, default=256, help="Batch size for training, controlling the number of samples per gradient update (default: 256)")
parser.add_argument('--epochs-autoencoder', type=int, default=200, help="Number of training epochs for the autoencoder (default: 200)")
parser.add_argument('--hidden-dim-encoder', type=int, default=256, help="Hidden dimension size for encoder layers (default: 64)")
parser.add_argument('--hidden-dim-stats-encoder', type=int, default=32, help="Hidden dimension size for encoder layers (default: 256)")
parser.add_argument('--hidden-dim-decoder', type=int, default=256, help="Hidden dimension size for decoder layers (default: 256)")
parser.add_argument('--latent-dim', type=int, default=2, help="Dimensionality of the latent space in the autoencoder (default: 32)")
parser.add_argument('--n-max-nodes', type=int, default=50, help="Possible maximum number of nodes in graphs (default: 50)")
parser.add_argument('--n-layers-encoder', type=int, default=3, help="Number of layers in the encoder network (default: 2)")
parser.add_argument('--n-layers-stats-encoder', type=int, default=3, help="Number of layers in the statistics encoder network (default: 3)")
parser.add_argument('--n-layers-decoder', type=int, default=3, help="Number of layers in the decoder network (default: 3)")
parser.add_argument('--spectral-emb-dim', type=int, default=10, help="Dimensionality of spectral embeddings for representing graph structures (default: 10)")
parser.add_argument('--n-condition', type=int, default=7, help="Number of distinct condition properties used in conditional vector (default: 7)")

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

trainset = preprocess_dataset('train', args.n_max_nodes, args.spectral_emb_dim)
dataset = preprocess_dataset(args.split, args.n_max_nodes, args.spectral_emb_dim)

if args.model_type == 'cvgae':
    stats_mean, stats_std = get_stats_mean_std(trainset)
    model = CVGAE(args.spectral_emb_dim+1, args.hidden_dim_encoder, args.hidden_dim_decoder, args.latent_dim, args.n_layers_encoder, args.n_layers_decoder, args.n_max_nodes, args.n_condition, stats_mean, stats_std).to(device)

if args.model_type == 'ngg':
    model = VariationalAutoEncoder(args.spectral_emb_dim+1, args.hidden_dim_encoder, args.hidden_dim_decoder, args.latent_dim, args.n_layers_encoder, args.n_layers_decoder, args.n_max_nodes).to(device)
    
if args.model_type == 'contrastive':
    stats_mean, stats_std = get_stats_mean_std(trainset)
    model = ContrastiveCVGAE(input_dim=args.spectral_emb_dim+1, hidden_dim_enc=args.hidden_dim_encoder, hidden_dim_dec=args.hidden_dim_decoder,hidden_dim_stats_enc=args.hidden_dim_stats_encoder, latent_dim=args.latent_dim, n_layers_enc=args.n_layers_encoder, n_layers_stats_enc=args.n_layers_stats_encoder, n_layers_dec=args.n_layers_decoder, n_max_nodes=args.n_max_nodes, n_cond=args.n_condition, stats_mean=stats_mean, stats_std=stats_std).to(device)

checkpoint = torch.load(args.model)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

if os.path.exists("visuals") == False:
    os.makedirs("visuals")

plot_latent(model, args.model_type, dataset, args.cond_idx, dim1 = 0, dim2 = 1, save_path = f"visuals/latent_{args.model_type}_{args.cond_idx}.png")
