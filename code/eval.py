import argparse
import re
import math

import networkx as nx
import numpy as np

import community as community_louvain

from utils import preprocess_dataset


np.random.seed(13)


parser = argparse.ArgumentParser(description='Configuration for the NeuralGraphGenerator model')
parser.add_argument('--file', type=str, default=None, help="CSV file path generated on the test set")
parser.add_argument('--spectral-emb-dim', type=int, default=10, help="Dimensionality of spectral embeddings for representing graph structures (default: 10)")
parser.add_argument('--n-max-nodes', type=int, default=50, help="Possible maximum number of nodes in graphs (default: 50)")

args = parser.parse_args()


testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim)
# test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)


def calculate_stats_graph(G):
    stats = []
    # Number of nodes
    num_nodes = handle_nan(float(G.number_of_nodes()))
    stats.append(num_nodes)
    # Number of edges
    num_edges = handle_nan(float(G.number_of_edges()))
    stats.append(num_edges)
    # Density

    # Degree statistics
    degrees = [deg for node, deg in G.degree()]

    avg_degree = handle_nan(float(sum(degrees) / len(degrees)))
    stats.append(avg_degree)
    # Assortativity coefficient

    # Number of triangles
    triangles = nx.triangles(G)
    num_triangles = handle_nan(float(sum(triangles.values()) // 3))
    stats.append(num_triangles)
    # Average number of triangles formed by an edge
    global_clustering_coefficient = handle_nan(float(nx.transitivity(G)))
    stats.append(global_clustering_coefficient)
    # Maximum k-core
    max_k_core = handle_nan(float(max(nx.core_number(G).values())))
    stats.append(max_k_core)
    # Lower bound of Maximum Clique
    #lower_bound_max_clique = handle_nan(float(nx.graph_clique_number(G)))
    #stats.append(lower_bound_max_clique)

    # calculate communities
    partition = community_louvain.best_partition(G)
    n_communities = handle_nan(float(len(set(partition.values()))))
    stats.append(n_communities)

    return stats


def gen_stats(G):
    y_pred = calculate_stats_graph(G)
    y_pred = np.nan_to_num(y_pred, nan=-100.0)
    return y_pred


def precompute_missing(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    y = np.nan_to_num(y, nan=-100.0)
    y_pred = np.nan_to_num(y_pred, nan=-100.0)
    # Find indices where y is -100
    indices_to_change = np.where(y == -100.0)

    # Set corresponding elements in y and y_pred to 0
    y[indices_to_change] = 0.0
    y_pred[indices_to_change] = 0.0
    zeros_per_column = np.count_nonzero(y, axis=0)

    list_from_array = zeros_per_column.tolist()
    dc = {}
    for i in range(len(list_from_array)):
        dc[i] = list_from_array[i]
    return dc, y, y_pred



def sum_elements_per_column(matrix, dc):
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    column_sums = [0] * num_cols

    for col in range(num_cols):
        for row in range(num_rows):
            column_sums[col] += matrix[row][col]

    res = []
    for col in range(num_cols):
        x = column_sums[col]/dc[col]
        res.append(x)

    return res



def calculate_mean_std(x):

    sm = [0 for i in range(7)]
    samples = [0 for i in range(7)]

    for el in x:
        for i, it in enumerate(el):
            if not math.isnan(it):
                sm[i] += it
                samples[i] += 1

    mean = [k / y for k,y in zip(sm, samples)]


    sm2 = [0 for i in range(16)]

    std = []

    for el in x:
        for i, it in enumerate(el):
            if not math.isnan(it):
                k = (it - mean[i])**2
                sm2[i] += k

    std = [(k / y)**0.5 for k,y in zip(sm2, samples)]
    return mean, std



def evaluation_metrics(y, y_pred, eps=1e-10):
    dc, y, y_pred = precompute_missing(y, y_pred)

    mse_st = (y - y_pred) ** 2
    mae_st = np.absolute(y - y_pred)

    mse = sum_elements_per_column(mse_st, dc)
    mae = sum_elements_per_column(mae_st, dc)

    #mse = [sum(x)/len(mse_st) for x in zip(*mse_st)]
    #mae = [sum(x)/len(mae_st) for x in zip(*mae_st)]

    a = np.absolute(y - y_pred)
    b = np.absolute(y) + np.absolute(y_pred)+ eps
    norm_error_st = (a/b)

    norm_error = sum_elements_per_column(norm_error_st, dc)
    #[sum(x)*100/len(norm_error_st) for x in zip(*norm_error_st)]

    return mse, mae, norm_error


def z_score_norm(y, y_pred, mean, std, eps=1e-10):

    y = np.array(y)
    y_pred = np.array(y_pred)

    normalized_true = (y - mean) / std

    normalized_gen = (y_pred - mean) / std

    dc, normalized_true, normalized_gen = precompute_missing(normalized_true, normalized_gen)

    #print(np.isnan(normalized_true).any())
    #print(np.isnan(normalized_gen).any())

    # Calculate MSE using normalized tensors
    mse_st = (normalized_true - normalized_gen) ** 2
    mae_st = np.absolute(normalized_true - normalized_gen)

    mse = sum_elements_per_column(mse_st, dc)
    mae = sum_elements_per_column(mae_st, dc)

    mse = np.sum(mse)/7
    mae = np.sum(mae)/7

    a = np.absolute(normalized_true - normalized_gen)
    b = np.absolute(normalized_true) + np.absolute(normalized_gen) + eps
    norm_error_st = (a/b)
    norm_error = sum_elements_per_column(norm_error_st, dc)
    norm_error = np.sum(norm_error)/7


    return mse, mae, norm_error


def construct_nx_from_adj(adj):
    G = nx.from_numpy_array(adj, create_using=nx.Graph)
    to_remove = []
    for node in G.nodes():
        if G.degree(node) == 0:
            to_remove.append(node)
    G.remove_nodes_from(to_remove)
    return G



def handle_nan(x):
    if math.isnan(x):
        return float(-100)
    return x


def read_stats(file):
    stats = []
    fread = open(file, "r")
    #print(file)
    for i,line in enumerate(fread):
        if i == 13: continue
        line = line.strip()
        tokens = line.split(":")
        #print(tokens[-1])
        #stats.append(handle_nan(float(tokens[-1].strip())))
        stats.append(float(tokens[-1].strip()))
    fread.close()
    return stats



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