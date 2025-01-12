import argparse
import re
import math

import networkx as nx
import numpy as np

import community as community_louvain

from utils import preprocess_dataset
from utils import gen_stats, calculate_mean_std, evaluation_metrics, z_score_norm

np.random.seed(13)


parser = argparse.ArgumentParser(description='Configuration for the NeuralGraphGenerator model')
parser.add_argument('--file', type=str, default=None, help="CSV file path generated on the test set")
parser.add_argument('--spectral-emb-dim', type=int, default=10, help="Dimensionality of spectral embeddings for representing graph structures (default: 10)")
parser.add_argument('--n-max-nodes', type=int, default=50, help="Possible maximum number of nodes in graphs (default: 50)")

args = parser.parse_args()


testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim)
# test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)






def main(args):
    ground_truth = []
    pred = []

    with open(args.file, "r") as csvfile:
        ground_truth = []
        pred = []
        line = csvfile.readline()
        for i in range(len(testset)):
            line = csvfile.readline()
            edges = re.findall(r'\((\d+), (\d+)\)', line)
            edge_list = [(int(u), int(v)) for u, v in edges]
            G = nx.Graph()
            G.add_edges_from(edge_list)
            stat_gt = testset.__getitem__(i).stats
            generated_stats  = np.array(gen_stats(G))
            
            ground_truth.append(stat_gt)
            pred.append(generated_stats)

    ground_truth, pred = np.array(ground_truth).squeeze(), np.array(pred).squeeze()

    mean, std = calculate_mean_std(ground_truth)
    mse, mae, norm_error = evaluation_metrics(ground_truth, pred)
    mse_all, mae_all, norm_error_all = z_score_norm(ground_truth, pred, mean, std)

    return {"mse" : mse, "mae" : mae, "norm_error" : norm_error, "mse_all" : mse_all, "mae_all": mae_all, "norm_error_all": norm_error_all}


if __name__=="__main__":
    res = main(args)
    print("Evaluation Results:")
    for key, value in res.items():
        print(f"{key}: {value}")