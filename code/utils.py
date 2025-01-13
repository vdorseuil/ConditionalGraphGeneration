import os
import torch
import networkx as nx
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sparse
from torch_geometric.data import Data
from tqdm import tqdm
from extract_feats import extract_feats

def preprocess_dataset(dataset, n_max_nodes, spectral_emb_dim, permute=False):
    """
    Preprocess a dataset ('train', 'test', or 'augment'), returning a list of Data objects.

    If dataset='augment', we iterate over multiple augmented subfolders (in 'augmented_list') 
    and create .pt files for each. We then combine all those augmented graphs into a single 
    data_lst and return it.

    To combine 'train' data and 'augment' data, you can do something like:

        train_data = preprocess_dataset('train', n_max_nodes, spectral_emb_dim)
        augment_data = preprocess_dataset('augment', n_max_nodes, spectral_emb_dim, permute=True)
        full_data = train_data + augment_data

    Then pass 'full_data' to your DataLoader.
    """

    data_lst = []

    # -------------------------------------------
    # 1) Handle 'test' case
    # -------------------------------------------
    if dataset == 'test':
        filename = './data/dataset_'+dataset+'.pt'
        desc_file = './data/'+dataset+'/test.txt'

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            print(f"Dataset {filename} loaded from file.")
        else:
            # Parse each line in test.txt
            with open(desc_file, "r") as fr:
                for line in fr:
                    line = line.strip()
                    tokens = line.split(",")
                    graph_id = tokens[0]
                    desc = tokens[1:]
                    desc = "".join(desc)
                    feats_stats = extract_numbers(desc)
                    feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)
                    data_lst.append(Data(stats=feats_stats, filename=graph_id))
            torch.save(data_lst, filename)
            print(f"Dataset {filename} saved.")
        return data_lst

    # -------------------------------------------
    # 2) Handle 'augment' case
    # -------------------------------------------
    elif dataset == 'augment':
        # (Key Change #1) We'll build a combined list of all augmented data
        combined_aug_data = []

        # Decide which augmented subfolders to process
        if permute: 
            augmented_list = ['adding_edges', 'adding_nodes', 'adding_triangles',
                              'changing_labels', 'changing_labels_bis']
        else: 
            augmented_list = ['adding_edges', 'adding_nodes', 'adding_triangles']

        # For each type in augmented_list, load or create the .pt file
        for datatype in augmented_list:
            filename = f'./data/dataset_{datatype}.pt'
            graph_path = f'./data_augmented/{datatype}/graph'
            desc_path = f'./data_augmented/{datatype}/description'

            if os.path.isfile(filename):
                # Already processed .pt for this augmented type
                partial_data = torch.load(filename)
                print(f"Augmented dataset {filename} loaded from file.")
            else:
                # We build partial_data from scratch
                partial_data = []
                files = [f for f in os.listdir(graph_path) if not f.startswith('.')]
                files.sort()
                for fileread in tqdm(files, desc=f"Processing {datatype}"):
                    tokens = fileread.split("/")
                    idx = tokens[-1].find(".")
                    filen = tokens[-1][:idx]
                    extension = tokens[-1][idx+1:]
                    fread = os.path.join(graph_path, fileread)
                    fstats = os.path.join(desc_path, filen + ".txt")

                    # Load dataset to networkx
                    if extension == "graphml":
                        G = nx.read_graphml(fread)
                        G = nx.convert_node_labels_to_integers(G, ordering="sorted")
                    else:
                        # Assuming edgelist
                        G = nx.read_edgelist(fread)

                    # BFS from largest-degree node (canonical order)
                    CGs = [G.subgraph(c) for c in nx.connected_components(G)]
                    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

                    node_list_bfs = []
                    for ii in range(len(CGs)):
                        node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
                        degree_sequence = sorted(node_degree_list, key=lambda tt: tt[1], reverse=True)
                        bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
                        node_list_bfs += list(bfs_tree.nodes())

                    adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)
                    adj = torch.from_numpy(adj_bfs).float()

                    # Build Laplacian
                    diags = np.sum(adj_bfs, axis=0).squeeze()
                    D = sparse.diags(diags).toarray()
                    L = D - adj_bfs
                    with np.errstate(divide="ignore"):
                        diags_sqrt = 1.0 / np.sqrt(diags)
                    diags_sqrt[np.isinf(diags_sqrt)] = 0
                    DH = sparse.diags(diags).toarray()
                    L = np.linalg.multi_dot((DH, L, DH))
                    L = torch.from_numpy(L).float()

                    eigval, eigvecs_val = torch.linalg.eigh(L)
                    eigval = torch.real(eigval)
                    eigvecs_val = torch.real(eigvecs_val)
                    idx_eigs = torch.argsort(eigval)
                    eigvecs_val = eigvecs_val[:, idx_eigs]

                    edge_index = torch.nonzero(adj).t()

                    size_diff = n_max_nodes - G.number_of_nodes()
                    x = torch.zeros(G.number_of_nodes(), spectral_emb_dim + 1)
                    if n_max_nodes > 1:  # to avoid div by zero
                        x[:, 0] = torch.mm(adj, torch.ones(G.number_of_nodes(), 1))[:, 0] / (n_max_nodes - 1)

                    mn = min(G.number_of_nodes(), spectral_emb_dim)
                    mn += 1
                    x[:, 1:mn] = eigvecs_val[:, :spectral_emb_dim]

                    # pad adjacency
                    adj = F.pad(adj, [0, size_diff, 0, size_diff])
                    adj = adj.unsqueeze(0)

                    feats_stats = extract_feats(fstats)
                    feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)

                    partial_data.append(Data(
                        x=x, 
                        edge_index=edge_index, 
                        A=adj, 
                        stats=feats_stats, 
                        filename=filen
                    ))

                # Save partial_data to a .pt
                torch.save(partial_data, filename)
                print(f"Augmented dataset {filename} saved.")

            # (Key Change #2) Combine partial_data from each augmentation
            combined_aug_data.extend(partial_data)

        # Return the combined list of *all* augmented data 
        return combined_aug_data

    # -------------------------------------------
    # 3) Handle 'train' or other sets
    # -------------------------------------------
    else:
        filename = './data/dataset_'+dataset+'.pt'
        graph_path = './data/'+dataset+'/graph'
        desc_path = './data/'+dataset+'/description'

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            print(f"Dataset {filename} loaded from file")
        else:
            # Build data_lst from scratch
            files = [f for f in os.listdir(graph_path) if not f.startswith('.')]
            files.sort()
            for fileread in tqdm(files, desc=f"Processing {dataset}"):
                tokens = fileread.split("/")
                idx = tokens[-1].find(".")
                filen = tokens[-1][:idx]
                extension = tokens[-1][idx+1:]
                fread = os.path.join(graph_path, fileread)
                fstats = os.path.join(desc_path, filen + ".txt")

                # Load to networkx
                if extension == "graphml":
                    G = nx.read_graphml(fread)
                    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
                else:
                    G = nx.read_edgelist(fread)

                # BFS from largest-degree node
                CGs = [G.subgraph(c) for c in nx.connected_components(G)]
                CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

                node_list_bfs = []
                for ii in range(len(CGs)):
                    node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
                    degree_sequence = sorted(node_degree_list, key=lambda tt: tt[1], reverse=True)
                    bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
                    node_list_bfs += list(bfs_tree.nodes())

                adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)
                adj = torch.from_numpy(adj_bfs).float()

                diags = np.sum(adj_bfs, axis=0).squeeze()
                D = sparse.diags(diags).toarray()
                L = D - adj_bfs
                with np.errstate(divide="ignore"):
                    diags_sqrt = 1.0 / np.sqrt(diags)
                diags_sqrt[np.isinf(diags_sqrt)] = 0
                DH = sparse.diags(diags).toarray()
                L = np.linalg.multi_dot((DH, L, DH))
                L = torch.from_numpy(L).float()

                eigval, eigvecs_val = torch.linalg.eigh(L)
                eigval = torch.real(eigval)
                eigvecs_val = torch.real(eigvecs_val)
                idx_eigs = torch.argsort(eigval)
                eigvecs_val = eigvecs_val[:, idx_eigs]

                edge_index = torch.nonzero(adj).t()

                size_diff = n_max_nodes - G.number_of_nodes()
                x = torch.zeros(G.number_of_nodes(), spectral_emb_dim+1)
                if n_max_nodes > 1:
                    x[:,0] = torch.mm(adj, torch.ones(G.number_of_nodes(), 1))[:,0]/(n_max_nodes-1)

                mn = min(G.number_of_nodes(), spectral_emb_dim)
                mn += 1
                x[:,1:mn] = eigvecs_val[:,:spectral_emb_dim]

                adj = F.pad(adj, [0, size_diff, 0, size_diff])
                adj = adj.unsqueeze(0)

                feats_stats = extract_feats(fstats)
                feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)

                data_lst.append(Data(
                    x=x, 
                    edge_index=edge_index, 
                    A=adj, 
                    stats=feats_stats, 
                    filename=filen
                ))

            torch.save(data_lst, filename)
            print(f"Dataset {filename} saved")

        return data_lst
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




def masked_instance_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = (torch.sum(x * mask, dim=[1,2]) / torch.sum(mask, dim=[1,2]))   # (N,C)
    var_term = ((x - mean.unsqueeze(1).unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[1,2]) / torch.sum(mask, dim=[1,2]))  # (N,C)
    mean = mean.unsqueeze(1).unsqueeze(1).expand_as(x)  # (N, L, L, C)
    var = var.unsqueeze(1).unsqueeze(1).expand_as(x)    # (N, L, L, C)
    instance_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    instance_norm = instance_norm * mask
    return instance_norm


def masked_layer_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = torch.sum(x * mask, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1])   # (N)
    var_term = ((x - mean.view(-1,1,1,1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1]))  # (N)
    mean = mean.view(-1,1,1,1).expand_as(x)  # (N, L, L, C)
    var = var.view(-1,1,1,1).expand_as(x)    # (N, L, L, C)
    layer_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    layer_norm = layer_norm * mask
    return layer_norm


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


