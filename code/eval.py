import argparse
import os
import random
import scipy as sp
import pickle
import sys
import re


import shutil
import csv
import ast
import community as community_louvain


import scipy.sparse as sparse
from tqdm import tqdm
from torch import Tensor
import networkx as nx
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch_geometric.data import Data

import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from autoencoder import VariationalAutoEncoder
from denoise_model import DenoiseNN, p_losses, sample
from utils import linear_beta_schedule, construct_nx_from_adj, preprocess_dataset, graph_statistics
from sklearn.metrics import mean_absolute_error
from scipy.stats import zscore

from torch.utils.data import Subset
np.random.seed(13)


parser = argparse.ArgumentParser(description='Configuration for the NeuralGraphGenerator model')
parser.add_argument('--file', type=str, default=None, help="CSV file path generated on the test set")
parser.add_argument('--spectral-emb-dim', type=int, default=10, help="Dimensionality of spectral embeddings for representing graph structures (default: 10)")
parser.add_argument('--n-max-nodes', type=int, default=50, help="Possible maximum number of nodes in graphs (default: 50)")

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
            generated_stats  = np.array(graph_statistics(G))
            
            ground_truth.append(stat_gt)
            pred.append(generated_stats)


    ground_truth = np.array(ground_truth)
    pred = np.array(pred)

    mean = np.nanmean(ground_truth, axis=0)
    std = np.nanstd(ground_truth, axis=0)

    z_scores_ground_truth = (ground_truth - mean) / std
    z_scores_pred = (pred - mean) / std
    mae = mean_absolute_error(z_scores_ground_truth.squeeze(), z_scores_pred.squeeze())
    return mae


if __name__=="__main__":
    print(f"\nmae : {main(args)}") 