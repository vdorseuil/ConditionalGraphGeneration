import argparse
import csv
import os

import numpy as np
import torch
from model.contrastive_cvae import ContrastiveCVGAE
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils.data_processing import (construct_nx_from_adj, get_stats_mean_std,
                                   preprocess_dataset)
from utils.eval import MAE, calculate_stats_graph, read_output

###############################################################################

# Argument parser
parser = argparse.ArgumentParser(description='NeuralGraphGenerator')


# Number of epochs for the autoencoder training
parser.add_argument('--epochs-autoencoder', type=int, default=200, help="Number of training epochs for the autoencoder (default: 200)")

# Hidden dimension size for the encoder network
parser.add_argument('--hidden-dim-encoder', type=int, default=256, help="Hidden dimension size for encoder layers (default: 256)")

# Hidden dimension size for the decoder network
parser.add_argument('--hidden-dim-decoder', type=int, default=256, help="Hidden dimension size for decoder layers (default: 256)")

# Hidden dimension size for the encoder network
parser.add_argument('--hidden-dim-stats-encoder', type=int, default=32, help="Hidden dimension size for encoder layers (default: 256)")

# Number of layers in the stats encoder network (contrastive learning)
parser.add_argument('--n-layers-stats-encoder', type=int, default=3, help="Number of layers in the statistics encoder network (default: 3)")

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

# Number of conditions used in conditional vector (number of properties)
parser.add_argument('--n-condition', type=int, default=7, help="Number of distinct condition properties used in conditional vector (default: 7)")

# Number of points to sample for the generation task, conditioned by the best MAE.
parser.add_argument('--n-sample', type=int, default=1, help="Number of points to sample in the VGAE latent space. The one with the best MAE is kept for the final generation. (default: 1)")

args = parser.parse_args()

###############################################################################


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

trainset = preprocess_dataset("train", args.n_max_nodes, args.spectral_emb_dim)
testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim)
test_loader = DataLoader(testset, batch_size=1, shuffle=False)

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

checkpoint = torch.load('./models/contrastive_cvgae.pth.tar')
contrastive_cvgae.load_state_dict(checkpoint['state_dict'])
contrastive_cvgae.eval()

if os.path.exists("outputs") == False:
    os.makedirs("outputs")

with open("outputs/output_CVGAE.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["graph_id", "edge_list"])


    for data in tqdm(test_loader, desc="Processing graphs"):
        data = data.to(device)
        stats = data.stats

        best_graph = None
        best_mae = np.inf
        for k in range(args.n_sample):
            adj = contrastive_cvgae.sample(stats)
            graph_ids = data.filename

            stat_x = stats[0] #batch size of 1
            Gs_generated = construct_nx_from_adj(adj[0,:,:].detach().cpu().numpy())
            test_stats = stat_x.detach().cpu().numpy()
            pred_stats = calculate_stats_graph(Gs_generated)
            mae =  np.mean(np.abs(pred_stats - test_stats))

            if mae < best_mae:
                best_mae = mae
                best_graph = Gs_generated

        # Define a graph ID
        graph_id = graph_ids[0]
        edge_list_text = ", ".join([f"({u}, {v})" for u, v in best_graph.edges()])           
        writer.writerow([graph_id, edge_list_text])


Gs = read_output("outputs/output_contrastive_CVGAE.csv")
mae_total = MAE(Gs, testset)
print(f"Mean absolute error of graph features: {mae_total:.2f}")