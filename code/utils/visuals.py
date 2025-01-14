import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn


condition_names = ['n_edges', 'n_nodes', 'avg_degree', 'n_triangles', 'global_clustering_coeff', 'max_k_core', 'n_communities']



def plot_latent(model, dataloader, cond_idx, dim1 = 0, dim2 = 1, save_path = 'visuals/latent_representations.png'):

    model.eval()
    zs = []
    stats = []

    with torch.no_grad():
        for data in dataloader:
            data = data.to(model.device)

            z, _, _ = model.encode(data)
            z = z.cpu().numpy()
            zs.append(z)

            stat = data.stats.cpu().numpy()
            stats.append(stat)

    zs = np.concatenate(zs, axis=0)
    stats = np.concatenate(stats, axis=0)

    plt.figure(figsize=(10, 10))
    plt.scatter(zs[:, dim1], zs[:, dim2], c=stats[:, cond_idx], cmap='viridis')
    plt.colorbar()
    plt.xlabel(f'z_{dim1}')
    plt.ylabel(f'z_{dim2}')
    plt.title(f'Latent representations vs condition {condition_names[cond_idx]}')

    plt.savefig(save_path)
    plt.show()