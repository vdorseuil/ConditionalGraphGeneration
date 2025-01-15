import os
import random

import networkx as nx
from parse_and_rewrite_description import parse_and_rewrite_description


def swap_random_nodes(G):
    """
    Swaps the labels of two distinct random nodes in-place, 
    if G has >= 2 nodes.
    """
    if G.number_of_nodes() < 2:
        return  # Can't swap if fewer than 2 nodes

    # Pick two distinct nodes at random
    a, b = random.sample(list(G.nodes()), 2)
    if a == b:
        return
    
    # We'll do a 3-step relabel to avoid collisions in-place:
    #  1) a -> temp_label
    #  2) b -> a
    #  3) temp_label -> b
    temp_label_candidate = -999999999
    while temp_label_candidate in G.nodes():
        temp_label_candidate -= 1

    # Step 1
    nx.relabel_nodes(G, {a: temp_label_candidate}, copy=False)
    # Step 2
    nx.relabel_nodes(G, {b: a}, copy=False)
    # Step 3
    nx.relabel_nodes(G, {temp_label_candidate: b}, copy=False)

def modify_graph_and_description(
    input_edge_file: str,
    input_description_file: str,
    output_edge_file: str,
    output_description_file: str,
    operation: str = "add_node",
    random_seed: int = 42,
    community_method: str = "louvain"
) -> None:
    """
    Reads a graph either from a GraphML file or a text edge list, modifies it
    randomly, and writes the new graph to either a GraphML file or a text edge
    list (depending on the `output_edge_file` extension).

    If an operation cannot be performed (e.g., the graph is full, or 
    too small to add an edge, etc.), we 'do nothing' structurally, 
    but instead swap labels of two nodes. That way, the graph 
    still changes in some way.

    Also updates the graph's metrics and rewrites the existing description file
    with those new metrics in the same style (using parse_and_rewrite_description).

    Parameters
    ----------
    input_edge_file : str
        Path to the file containing the original graph (could be .graphml or .txt).
    input_description_file : str
        Path to the file containing the original description (only read for rewriting).
    output_edge_file : str
        Path to write the modified graph. If it ends with .graphml, we output GraphML;
        otherwise, we write an edge list.
    output_description_file : str
        Path to write the new graph description.
    operation : str
        One of:
          - "add_node": add a new node with no edges
          - "add_edge": add an edge between two existing nodes
          - "add_node_and_edge": add a new node and connect it to an existing node
          - "add_triangle": add exactly one edge to form a new triangle (fallback: 2 edges)
    random_seed : int, optional
        Seed for reproducibility. If None, no seed is set.
    community_method : str, optional
        Which community detection method to use (e.g., "louvain", "greedy", "label").
    """

    # Optionally set random seed
    if random_seed is not None:
        random.seed(random_seed)

    # 1. Read the original graph
    in_ext = os.path.splitext(input_edge_file)[1].lower()
    
    if in_ext == ".graphml":
        G = nx.read_graphml(input_edge_file)
        # Convert string labels to int if numeric
        try:
            G_int = nx.convert_node_labels_to_integers(G, ordering="sorted")
            G = G_int
        except Exception:
            pass
    else:
        G = nx.Graph()
        with open(input_edge_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    u = int(parts[0])
                    v = int(parts[1])
                    G.add_edge(u, v)

    do_nothing = False  # we'll track if we fail to do the main operation

    # 2. Perform the requested operation
    if operation == "add_node":
        # Only add if total nodes < 50
        if G.number_of_nodes() < 50:
            new_node = max(G.nodes) + 1 if G.number_of_nodes() > 0 else 0
            G.add_node(new_node)
        else:
            do_nothing = True

    elif operation == "add_edge":
        # Attempt to add an edge
        nodes_list = list(G.nodes())
        if len(nodes_list) < 2:
            do_nothing = True
        else:
            possible_new_edge = None
            for _ in range(1000):
                u = random.choice(nodes_list)
                v = random.choice(nodes_list)
                if u != v and not G.has_edge(u, v):
                    possible_new_edge = (u, v)
                    break
            if possible_new_edge is not None:
                G.add_edge(*possible_new_edge)
            else:
                # Graph might be complete
                do_nothing = True

    elif operation == "add_node_and_edge":
        # Add a new node + one edge to a random existing node
        if G.number_of_nodes() >= 50:
            do_nothing = True
        else:
            if G.number_of_nodes() == 0:
                # If empty, just add node 0
                G.add_node(0)
            else:
                new_node = max(G.nodes) + 1
                existing_node = random.choice(list(G.nodes()))
                G.add_node(new_node)
                G.add_edge(new_node, existing_node)

    elif operation == "add_triangle":
        # Attempt to add exactly one edge that forms a new triangle
        if G.number_of_nodes() < 3:
            do_nothing = True
        else:
            created_triangle = False
            node_list = list(G.nodes())
            for _ in range(1000):
                w = random.choice(node_list)
                neighbors_w = list(G.neighbors(w))
                if len(neighbors_w) < 2:
                    continue
                x, y = random.sample(neighbors_w, 2)
                if not G.has_edge(x, y):
                    G.add_edge(x, y)
                    created_triangle = True
                    break

            if not created_triangle:
                # Try adding two edges
                edges_added = 0
                attempts = 0
                while edges_added < 2 and attempts < 2000:
                    attempts += 1
                    u = random.choice(node_list)
                    v = random.choice(node_list)
                    if u != v and not G.has_edge(u, v):
                        G.add_edge(u, v)
                        edges_added += 1
                if edges_added < 2:
                    do_nothing = True

    else:
        # Unknown operation => do nothing
        raise ValueError(f"Unsupported operation: {operation}")

    # If we ended up doing nothing, swap the labels of two nodes
    if do_nothing:
        swap_random_nodes(G)

    # 3. Write out the modified graph
    out_ext = os.path.splitext(output_edge_file)[1].lower()
    if out_ext == ".graphml":
        nx.write_graphml(G, output_edge_file)
    else:
        with open(output_edge_file, "w") as f_out:
            for u, v in sorted(G.edges()):
                f_out.write(f"{u} {v} {{}}\n")

    # 4. Compute updated graph metrics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    average_degree = (2.0 * num_edges / num_nodes) if num_nodes > 0 else 0.0

    # Count triangles
    triangle_dict = nx.triangles(G)
    total_triangles = sum(triangle_dict.values()) // 3

    # Global clustering coefficient
    global_clustering = nx.transitivity(G)

    # Max k-core
    k = 0
    while True:
        core_subgraph = nx.k_core(G, k=k)
        if core_subgraph.number_of_nodes() == 0:
            max_k_core = k - 1 if k > 0 else 0
            break
        k += 1

    # Number of communities
    if community_method == "louvain":
        # Using networkx's built-in Louvain method for community detection
        communities = nx.community.louvain_communities(G, seed=123)
        num_communities = len(communities)
    elif community_method == "greedy":
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        num_communities = len(communities)
    elif community_method == "label":
        from networkx.algorithms.community import label_propagation_communities
        communities = list(label_propagation_communities(G))
        num_communities = len(communities)
    else:
        num_communities = 1

    # 5. Build & write the new description
    new_values = [
        num_nodes,
        num_edges,
        average_degree,
        total_triangles,
        global_clustering,
        max_k_core,
        num_communities
    ]
    with open(input_description_file, 'r', encoding='utf-8') as infile:
        original_description = infile.read()

    parse_and_rewrite_description(
        input_description_str=original_description,
        new_values=new_values,
        output_path=output_description_file
    )
    # Done, no return needed
