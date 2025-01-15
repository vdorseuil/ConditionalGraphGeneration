import os

from generate_permuted_graphs import generate_permuted_graphs
from modify_graph_and_description import modify_graph_and_description
from tqdm import tqdm

if __name__ == "__main__":
    # 1. Base path
    base_path = '/Users/theo/MVA/Altegrad/Final_project/altegrad_project'
    
    # 2. Build subpaths for 'train' data
    graph_path = os.path.join(base_path, "data", "train", "graph")
    description_path = os.path.join(base_path, "data", "train", "description")

    # 3. Folders for augmented data
    data_augmented_folder = os.path.join(base_path, "data_augmented")

    output_adding_edges_graph = os.path.join(data_augmented_folder, "adding_edges", "graph")
    output_adding_edges_desc = os.path.join(data_augmented_folder, "adding_edges", "description")

    output_adding_triangles_graph = os.path.join(data_augmented_folder, "adding_triangles", "graph")
    output_adding_triangles_desc = os.path.join(data_augmented_folder, "adding_triangles", "description")

    # NEW FOLDER FOR ADDING NODES
    output_adding_nodes_graph = os.path.join(data_augmented_folder, "adding_nodes", "graph")
    output_adding_nodes_desc = os.path.join(data_augmented_folder, "adding_nodes", "description")

    # Folders for "changing labels"
    output_changing_labels_graph = os.path.join(data_augmented_folder, "changing_labels", "graph")
    output_changing_labels_desc = os.path.join(data_augmented_folder, "changing_labels", "description")

    output_changing_labels_bis_graph = os.path.join(data_augmented_folder, "changing_labels_bis", "graph")
    output_changing_labels_bis_desc = os.path.join(data_augmented_folder, "changing_labels_bis", "description")

    # 4. Create those folders if they don't exist
    os.makedirs(data_augmented_folder, exist_ok=True)
    os.makedirs(output_adding_edges_graph, exist_ok=True)
    os.makedirs(output_adding_edges_desc, exist_ok=True)
    os.makedirs(output_adding_triangles_graph, exist_ok=True)
    os.makedirs(output_adding_triangles_desc, exist_ok=True)
    os.makedirs(output_adding_nodes_graph, exist_ok=True)
    os.makedirs(output_adding_nodes_desc, exist_ok=True)
    os.makedirs(output_changing_labels_graph, exist_ok=True)
    os.makedirs(output_changing_labels_desc, exist_ok=True)
    os.makedirs(output_changing_labels_bis_graph, exist_ok=True)
    os.makedirs(output_changing_labels_bis_desc, exist_ok=True)

    # 5. Gather list of graph files (train graphs)
    files = [f for f in os.listdir(graph_path) if not f.startswith(".")]
    files.sort()  # Optional: consistent ordering

    for i in tqdm(range(8000), desc='augmenting data'):
        # The original graph file (e.g. "graph_0.edgelist")
        edge_file_name = files[i]
        edge_file = os.path.join(graph_path, edge_file_name)

        # The corresponding original description file
        description_file = os.path.join(description_path, f"graph_{i}.txt")

        # Determine extension
        _, file_ext = os.path.splitext(edge_file_name)
        file_ext = file_ext.lower()
        if file_ext == ".graphml":
            out_ext = ".graphml"
        else:
            out_ext = ".edgelist"

        ############################################################################
        # A) Add Edges
        ############################################################################
        file_path_graph_edge = os.path.join(output_adding_edges_graph, f"graph_{i}{out_ext}")
        file_path_desc_edge = os.path.join(output_adding_edges_desc, f"graph_{i}.txt")

        with open(file_path_graph_edge, "w"):
            pass
        with open(file_path_desc_edge, "w"):
            pass

        modify_graph_and_description(
            input_edge_file=edge_file,
            input_description_file=description_file,
            output_edge_file=file_path_graph_edge,
            output_description_file=file_path_desc_edge,
            operation="add_edge"
        )

        ############################################################################
        # B) Add Triangles
        ############################################################################
        file_path_graph_triangle = os.path.join(output_adding_triangles_graph, f"graph_{i}{out_ext}")
        file_path_desc_triangle = os.path.join(output_adding_triangles_desc, f"graph_{i}.txt")

        with open(file_path_graph_triangle, "w"):
            pass
        with open(file_path_desc_triangle, "w"):
            pass

        modify_graph_and_description(
            input_edge_file=edge_file,
            input_description_file=description_file,
            output_edge_file=file_path_graph_triangle,
            output_description_file=file_path_desc_triangle,
            operation="add_triangle"
        )

        ############################################################################
        # C) Add Node
        ############################################################################
        file_path_graph_node = os.path.join(output_adding_nodes_graph, f"graph_{i}{out_ext}")
        file_path_desc_node = os.path.join(output_adding_nodes_desc, f"graph_{i}.txt")

        with open(file_path_graph_node, "w"):
            pass
        with open(file_path_desc_node, "w"):
            pass

        modify_graph_and_description(
            input_edge_file=edge_file,
            input_description_file=description_file,
            output_edge_file=file_path_graph_node,
            output_description_file=file_path_desc_node,
            operation="add_node_and_edge"
        )

        ############################################################################
        # D) Change labels (permutations) on the ORIGINAL graph
        #    => two permuted outputs
        ############################################################################
        file_path_graph = os.path.join(output_changing_labels_graph, f"graph_{4*i}{out_ext}")
        file_path_desc = os.path.join(output_changing_labels_desc, f"graph_{4*i}.txt")

        file_path_graph_bis = os.path.join(output_changing_labels_bis_graph, f"graph_{4*i}{out_ext}")
        file_path_desc_bis = os.path.join(output_changing_labels_bis_desc, f"graph_{4*i}.txt")

        with open(file_path_graph, 'w'):
            pass
        with open(file_path_graph_bis, 'w'):
            pass

        # Copy original description into both
        with open(description_file, 'r', encoding='utf-8') as f_in:
            content_original_desc = f_in.read()
        with open(file_path_desc, 'w', encoding='utf-8') as f_out:
            f_out.write(content_original_desc)
        with open(file_path_desc_bis, 'w', encoding='utf-8') as f_out:
            f_out.write(content_original_desc)

        # Generate permutations from the ORIGINAL
        generate_permuted_graphs(
            input_edge_file=edge_file,
            output_file_1=file_path_graph,
            output_file_2=file_path_graph_bis
        )

        ############################################################################
        # E) Change labels for the newly modified graph (Add Edge)
        ############################################################################
        file_path_graph_edge_perm = os.path.join(output_changing_labels_graph, f"graph_{4*i+1}{out_ext}")
        file_path_desc_edge_perm = os.path.join(output_changing_labels_desc, f"graph_{4*i+1}.txt")

        file_path_graph_edge_perm_bis = os.path.join(output_changing_labels_bis_graph, f"graph_{4*i+1}{out_ext}")
        file_path_desc_edge_perm_bis = os.path.join(output_changing_labels_bis_desc, f"graph_{4*i+1}.txt")

        with open(file_path_graph_edge_perm, 'w'):
            pass
        with open(file_path_graph_edge_perm_bis, 'w'):
            pass

        # Copy the *modified* (add_edge) description
        with open(file_path_desc_edge, 'r', encoding='utf-8') as f_in:
            content_edge_desc = f_in.read()
        with open(file_path_desc_edge_perm, 'w', encoding='utf-8') as f_out:
            f_out.write(content_edge_desc)
        with open(file_path_desc_edge_perm_bis, 'w', encoding='utf-8') as f_out:
            f_out.write(content_edge_desc)

        # Generate permutations from the newly created add_edge graph
        generate_permuted_graphs(
            input_edge_file=file_path_graph_edge,  # the newly created graph
            output_file_1=file_path_graph_edge_perm,
            output_file_2=file_path_graph_edge_perm_bis
        )

        ############################################################################
        # F) Change labels for the newly modified graph (Add Triangle)
        ############################################################################
        file_path_graph_tri_perm = os.path.join(output_changing_labels_graph, f"graph_{4*i+2}{out_ext}")
        file_path_desc_tri_perm = os.path.join(output_changing_labels_desc, f"graph_{4*i+2}.txt")

        file_path_graph_tri_perm_bis = os.path.join(output_changing_labels_bis_graph, f"graph_{4*i+2}{out_ext}")
        file_path_desc_tri_perm_bis = os.path.join(output_changing_labels_bis_desc, f"graph_{4*i+2}.txt")

        with open(file_path_graph_tri_perm, 'w'):
            pass
        with open(file_path_graph_tri_perm_bis, 'w'):
            pass

        # Copy the *modified* (add_triangle) description
        with open(file_path_desc_triangle, 'r', encoding='utf-8') as f_in:
            content_triangle_desc = f_in.read()
        with open(file_path_desc_tri_perm, 'w', encoding='utf-8') as f_out:
            f_out.write(content_triangle_desc)
        with open(file_path_desc_tri_perm_bis, 'w', encoding='utf-8') as f_out:
            f_out.write(content_triangle_desc)

        generate_permuted_graphs(
            input_edge_file=file_path_graph_triangle,
            output_file_1=file_path_graph_tri_perm,
            output_file_2=file_path_graph_tri_perm_bis
        )

        ############################################################################
        # G) Change labels for the newly modified graph (Add Node)
        ############################################################################
        file_path_graph_node_perm = os.path.join(output_changing_labels_graph, f"graph_{4*i+3}{out_ext}")
        file_path_desc_node_perm = os.path.join(output_changing_labels_desc, f"graph_{4*i+3}.txt")

        file_path_graph_node_perm_bis = os.path.join(output_changing_labels_bis_graph, f"graph_{4*i+3}{out_ext}")
        file_path_desc_node_perm_bis = os.path.join(output_changing_labels_bis_desc, f"graph_{4*i+3}.txt")

        with open(file_path_graph_node_perm, 'w'):
            pass
        with open(file_path_graph_node_perm_bis, 'w'):
            pass

        # Copy the *modified* (add_node) description
        with open(file_path_desc_node, 'r', encoding='utf-8') as f_in:
            content_node_desc = f_in.read()
        with open(file_path_desc_node_perm, 'w', encoding='utf-8') as f_out:
            f_out.write(content_node_desc)
        with open(file_path_desc_node_perm_bis, 'w', encoding='utf-8') as f_out:
            f_out.write(content_node_desc)

        # Generate permutations from the newly created add_node graph
        generate_permuted_graphs(
            input_edge_file=file_path_graph_node,
            output_file_1=file_path_graph_node_perm,
            output_file_2=file_path_graph_node_perm_bis
        )

    print("Data augmentation complete!")
