import argparse
import csv
import os

import numpy as np
import torch
from model.cvae import CVGAE
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils.data_processing import (construct_nx_from_adj, get_stats_mean_std,
                                   preprocess_dataset)
from utils.eval import MAE, calculate_stats_graph, read_output

###############################################################################

# Argument parser
parser = argparse.ArgumentParser(description='NeuralGraphGenerator')

parser.add_argument('--epochs-autoencoder', type=int, default=200, help="Number of training epochs for the autoencoder (default: 200)")
parser.add_argument('--hidden-dim-encoder', type=int, default=256, help="Hidden dimension size for encoder layers (default: 64)")
parser.add_argument('--hidden-dim-decoder', type=int, default=256, help="Hidden dimension size for decoder layers (default: 256)")
parser.add_argument('--latent-dim', type=int, default=2, help="Dimensionality of the latent space in the autoencoder (default: 32)")
parser.add_argument('--n-max-nodes', type=int, default=50, help="Possible maximum number of nodes in graphs (default: 50)")
parser.add_argument('--n-layers-encoder', type=int, default=3, help="Number of layers in the encoder network (default: 2)")
parser.add_argument('--n-layers-decoder', type=int, default=3, help="Number of layers in the decoder network (default: 3)")
parser.add_argument('--spectral-emb-dim', type=int, default=10, help="Dimensionality of spectral embeddings for representing graph structures (default: 10)")
parser.add_argument('--n-condition', type=int, default=7, help="Number of distinct condition properties used in conditional vector (default: 7)")
parser.add_argument('--n-sample', type=int, default=1, help="Number of points to sample in the VGAE latent space. The one with the best MAE is kept for the final generation. (default: 1)")

args = parser.parse_args()

###############################################################################


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

trainset = preprocess_dataset("train", args.n_max_nodes, args.spectral_emb_dim)
testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim)
test_loader = DataLoader(testset, batch_size=1, shuffle=False)

stats_mean, stats_std = get_stats_mean_std(trainset)

cvgae = CVGAE(args.spectral_emb_dim+1, args.hidden_dim_encoder, args.hidden_dim_decoder, args.latent_dim, args.n_layers_encoder, args.n_layers_decoder, args.n_max_nodes, args.n_condition, stats_mean, stats_std).to(device)
checkpoint = torch.load('./models/cvgae.pth.tar')
cvgae.load_state_dict(checkpoint['state_dict'])
cvgae.eval()

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
            adj = cvgae.sample(stats)
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

Gs = read_output("outputs/output_CVGAE.csv")
mae_total = MAE(Gs, testset)
print(f"Mean absolute error of graph features: {mae_total:.2f}")