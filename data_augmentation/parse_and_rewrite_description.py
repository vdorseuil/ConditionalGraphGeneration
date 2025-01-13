import re

def parse_and_rewrite_description(
    input_description_str: str,
    new_values: list[float],
    output_path: str,
) -> None:
    """
    Detects which of the two description formats (Type A or Type B) is being used,
    extracts the 7 numbers, and rewrites a new description with the same wording
    but updated numbers. The new description is written to `output_path`.

    Parameters
    ----------
    input_description_str : str
        The original description string (either Type A or Type B).
    new_values : list[float]
        A list of 7 new numeric values to replace in the description.
        The order should be:
          [num_nodes, num_edges, avg_degree, num_triangles,
           global_clustering, max_k_core, num_communities]
    output_path : str
        Path where the rewritten description will be saved.

    Returns
    -------
    None
    """

    # We expect exactly 7 new numeric values.
    if len(new_values) != 7:
        raise ValueError("new_values must contain exactly 7 numbers.")

    # --- REGEX for Type A ---
    # Example:
    # "This graph comprises 19 nodes and 152 edges. 
    #  The average degree is equal to 16.0 and there are 665 triangles in the graph. 
    #  The global clustering coefficient and the graph's maximum k-core are 0.875 and 16 respectively. 
    #  The graph consists of 2 communities."
    # We'll capture groups in the order we want them:
    #   1) nodes
    #   2) edges
    #   3) average_degree
    #   4) triangles
    #   5) global_clustering
    #   6) max_k_core
    #   7) communities
    type_a_pattern = re.compile(
        r"^This graph comprises\s+(\d+)\s+nodes and\s+(\d+)\s+edges\.\s+"
        r"The average degree is equal to\s+([\d\.]+)\s+and there are\s+(\d+)\s+triangles in the graph\.\s+"
        r"The global clustering coefficient and the graph's maximum k-core are\s+([\d\.]+)\s+and\s+(\d+)\s+respectively\.\s+"
        r"The graph consists of\s+(\d+)\s+communities\.$"
    )

    # --- REGEX for Type B ---
    # Example:
    # "In this graph, there are 25 nodes connected by 147 edges. 
    #  On average, each node is connected to 11.76 other nodes. 
    #  Within the graph, there are 269 triangles, forming closed loops of nodes. 
    #  The global clustering coefficient is 0.4890909090909091. 
    #  Additionally, the graph has a maximum k-core of 8 and a number of communities equal to 4."
    type_b_pattern = re.compile(
        r"^In this graph, there are\s+(\d+)\s+nodes connected by\s+(\d+)\s+edges\.\s+"
        r"On average, each node is connected to\s+([\d\.]+)\s+other nodes\.\s+"
        r"Within the graph, there are\s+(\d+)\s+triangles, forming closed loops of nodes\.\s+"
        r"The global clustering coefficient is\s+([\d\.]+)\.\s+"
        r"Additionally, the graph has a maximum k-core of\s+(\d+)\s+and a number of communities equal to\s+(\d+)\.$"
    )

    # Try matching Type A
    match_a = type_a_pattern.match(input_description_str.strip())
    if match_a:
        # We found it is Type A
        # Extract the numeric values from the match
        old_values = list(match_a.groups())  # strings
        # Convert to float or int
        # Indices for Type A:
        #   0 -> nodes (int)
        #   1 -> edges (int)
        #   2 -> avg_degree (float)
        #   3 -> triangles (int)
        #   4 -> global_clustering (float)
        #   5 -> max_k_core (int)
        #   6 -> communities (int)
        old_values_numeric = [
            int(old_values[0]),
            int(old_values[1]),
            float(old_values[2]),
            int(old_values[3]),
            float(old_values[4]),
            int(old_values[5]),
            int(old_values[6])
        ]

        # Just to illustrate, we put them in a vector (not strictly required):
        old_values_vector = old_values_numeric
        # new_values_vector is the 7-element list passed in new_values
        new_nodes, new_edges, new_avg_deg, new_triangles, new_gcc, new_kcore, new_comms = new_values

        # Build the new description with exactly the same wording:
        new_description = (
            f"This graph comprises {int(new_nodes)} nodes and {int(new_edges)} edges. "
            f"The average degree is equal to {new_avg_deg} and there are {int(new_triangles)} triangles in the graph. "
            f"The global clustering coefficient and the graph's maximum k-core are {new_gcc} and {int(new_kcore)} respectively. "
            f"The graph consists of {int(new_comms)} communities."
        )

        # Write out to file
        with open(output_path, 'w') as f:
            f.write(new_description)

        return  # Done

    # Try matching Type B
    match_b = type_b_pattern.match(input_description_str.strip())
    if match_b:
        # We found it is Type B
        # Extract the numeric values from the match
        old_values = list(match_b.groups())  # strings
        # Convert to float or int
        # Indices for Type B:
        #   0 -> nodes (int)
        #   1 -> edges (int)
        #   2 -> avg_degree (float)
        #   3 -> triangles (int)
        #   4 -> global_clustering (float)
        #   5 -> max_k_core (int)
        #   6 -> communities (int)
        old_values_numeric = [
            int(old_values[0]),
            int(old_values[1]),
            float(old_values[2]),
            int(old_values[3]),
            float(old_values[4]),
            int(old_values[5]),
            int(old_values[6])
        ]

        # Same approach as Type A
        new_nodes, new_edges, new_avg_deg, new_triangles, new_gcc, new_kcore, new_comms = new_values

        # Build the new description with exactly the same wording:
        new_description = (
            f"In this graph, there are {int(new_nodes)} nodes connected by {int(new_edges)} edges. "
            f"On average, each node is connected to {new_avg_deg} other nodes. "
            f"Within the graph, there are {int(new_triangles)} triangles, forming closed loops of nodes. "
            f"The global clustering coefficient is {new_gcc}. "
            f"Additionally, the graph has a maximum k-core of {int(new_kcore)} and a number of communities equal to {int(new_comms)}."
        )

        # Write out to file
        with open(output_path, 'w') as f:
            f.write(new_description)

        return  # Done

    # If neither pattern matches, raise an error
    raise ValueError("Input description did not match either Type A or Type B format.")


