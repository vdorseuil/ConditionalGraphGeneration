from torch_geometric.loader import DataLoader
import torch
import csv
import argparse
import os

from model.cvae import CVGAE
from utils.data_processing import preprocess_dataset, construct_nx_from_adj, get_stats_mean_std
from utils.eval import read_output, MAE

###############################################################################

# Argument parser
parser = argparse.ArgumentParser(description='NeuralGraphGenerator')

# Batch size
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

# Number of conditions used in conditional vector (number of properties)
parser.add_argument('--n-condition', type=int, default=7, help="Number of distinct condition properties used in conditional vector (default: 7)")

args = parser.parse_args()

###############################################################################


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

trainset = preprocess_dataset("train", args.n_max_nodes, args.spectral_emb_dim)
testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

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


    for data in test_loader:
        data = data.to(device)
        stats = data.stats
        adj = cvgae.sample(stats)
        graph_ids = data.filename


        for i in range(stats.size(0)):
            stat_x = stats[i]

            Gs_generated = construct_nx_from_adj(adj[i,:,:].detach().cpu().numpy())
            stat_x = stat_x.detach().cpu().numpy()

            # Define a graph ID
            graph_id = graph_ids[i]

            # Convert the edge list to a single string
            edge_list_text = ", ".join([f"({u}, {v})" for u, v in Gs_generated.edges()])           
            # Write the graph ID and the full edge list as a single row
            writer.writerow([graph_id, edge_list_text])


Gs = read_output("outputs/output_CVGAE.csv")
mae = MAE(Gs, testset)
print(f"Mean absolute error of graph features: {mae:.2f}")