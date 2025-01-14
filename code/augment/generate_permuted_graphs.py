import os
import random
import networkx as nx

def generate_permuted_graphs(
    input_edge_file: str,
    output_file_1: str,
    output_file_2: str
) -> None:
    """
    Reads a graph either from a GraphML file or an edge-list file, then
    produces two new graph files (output_file_1, output_file_2) each describing
    the same graph but with different permutations (relabeling) of nodes.

    If the input file has a .graphml extension, we read/write GraphML.
    Otherwise, we assume edge-list format (each line: "u v" or "u v {}") and
    write the same format to the output files.

    Parameters
    ----------
    input_edge_file : str
        Path to the file containing the original graph. Could be .graphml or
        a text edge list.
    output_file_1 : str
        Path to the file where the 1st permuted graph will be written.
        The extension decides if we write GraphML or a plain edge list.
    output_file_2 : str
        Path to the file where the 2nd permuted graph will be written.
        Same logic as above.
    """

    # Decide if we're dealing with GraphML or edge list based on extension
    _, in_ext = os.path.splitext(input_edge_file)
    in_ext = in_ext.lower()

    # Function to create a node permutation mapping given a NetworkX graph
    def create_permutation_mappings(G):
        original_nodes = sorted(G.nodes())
        random.seed(42)  # For reproducibility
        permuted_nodes_1 = original_nodes[:]
        permuted_nodes_2 = original_nodes[:]
        random.shuffle(permuted_nodes_1)
        random.shuffle(permuted_nodes_2)
        # Build dict: old_label -> new_label
        perm_map_1 = {}
        perm_map_2 = {}
        for old, new1, new2 in zip(original_nodes, permuted_nodes_1, permuted_nodes_2):
            perm_map_1[old] = new1
            perm_map_2[old] = new2
        return perm_map_1, perm_map_2

    # ---------------------------------------------------------------------
    # 1. Read the graph (GraphML or edge-list)
    # ---------------------------------------------------------------------
    if in_ext == ".graphml":
        # Load as GraphML
        G = nx.read_graphml(input_edge_file)

        # Convert any non-integer labels (strings) to int if needed
        # (If your GraphML is guaranteed to have numeric labels,
        #  you might skip convert_node_labels_to_integers.)
        try:
            # Try to interpret the labels as ints; if it fails, we'll just keep them as-is
            G_int = nx.convert_node_labels_to_integers(G, ordering="sorted")
            G = G_int
        except Exception:
            pass

    else:
        # Assume text edge list
        G = nx.Graph()
        with open(input_edge_file, "r") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                # Convert to int if your node labels are integers
                u = int(parts[0])
                v = int(parts[1])
                G.add_edge(u, v)

    # ---------------------------------------------------------------------
    # 2. Generate the two permutations
    # ---------------------------------------------------------------------
    perm_map_1, perm_map_2 = create_permutation_mappings(G)

    # ---------------------------------------------------------------------
    # 3. Create the two new graphs by relabeling
    # ---------------------------------------------------------------------
    # NetworkX has a built-in relabel_nodes function
    G_perm_1 = nx.relabel_nodes(G, perm_map_1)
    G_perm_2 = nx.relabel_nodes(G, perm_map_2)

    # ---------------------------------------------------------------------
    # 4. Write them out (GraphML if input was GraphML, else edge list)
    # ---------------------------------------------------------------------
    _, out_ext_1 = os.path.splitext(output_file_1)
    out_ext_1 = out_ext_1.lower()

    _, out_ext_2 = os.path.splitext(output_file_2)
    out_ext_2 = out_ext_2.lower()

    # Decide how to write each graph. Typically you'd keep the same format
    # as the input, but if you specifically want to enforce GraphML, do so:
    if in_ext == ".graphml":
        # Write GraphML
        nx.write_graphml(G_perm_1, output_file_1)
        nx.write_graphml(G_perm_2, output_file_2)
    else:
        # Write edge list
        # Keep the same format "u v {}" or just "u v"
        with open(output_file_1, "w") as f1:
            for (u, v) in G_perm_1.edges():
                f1.write(f"{u} {v} {{}}\n")

        with open(output_file_2, "w") as f2:
            for (u, v) in G_perm_2.edges():
                f2.write(f"{u} {v} {{}}\n")
