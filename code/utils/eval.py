import networkx as nx
import numpy as np
import csv
import community as community_louvain
import ast


def calculate_stats_graph(G):
    stats = np.zeros(7)
    stats[0] = G.number_of_nodes()
    stats[1] = G.number_of_edges()
    stats[2] = np.mean(list(dict(G.degree()).values()))
    stats[3] = sum(list(nx.triangles(G).values()))/3
    stats[4] = nx.transitivity(G)
    stats[5] = max(nx.core_number(G).values())
    stats[6] = len(set(community_louvain.best_partition(G).values()))
    return stats

def read_output(output_file):
    graphs = []
    with open(output_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            edge_list_str = row[1]
            edge_list = ast.literal_eval(edge_list_str)

            graph = nx.Graph()
            graph.add_edges_from(edge_list)

            graphs.append(graph)

    return graphs



def MAE(pred_Gs, testset):
    mae = 0
    for i in range(len(pred_Gs)):
        pred_G = pred_Gs[i]
        test_stats = testset[i].stats.cpu().numpy()

        pred_stats = calculate_stats_graph(pred_G)

        mae += np.mean(np.abs(pred_stats - test_stats))
    return mae/len(pred_Gs)
    

