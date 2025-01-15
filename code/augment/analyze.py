import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


###############################################################################
# 1. PARSING THE 7 NUMERICAL VALUES (placeholder)
###############################################################################
def extract_7_numbers(description_str: str) -> list[float]:
    """
    Quick approach to parse 7 numeric features:
      [num_nodes, num_edges, avg_degree, num_triangles,
       global_clustering, max_k_core, num_communities].
    Adjust to your exact description format if needed.
    """
    nums = re.findall(r"\d+\.\d+|\d+", description_str)
    # Convert first 7 to float
    values = [float(x) for x in nums[:7]]
    return values

###############################################################################
# 2. GATHER DATA FROM MULTIPLE FOLDERS
###############################################################################
def collect_description_features(folders_dict):
    """
    Reads .txt files from each folder in `folders_dict` and extracts the 7 metrics.
    Returns a DataFrame with columns:
      [num_nodes, num_edges, avg_degree, num_triangles,
       global_clustering, max_k_core, num_communities, data_type].
    """
    data_rows = []
    for label, folder_path in folders_dict.items():
        desc_files = [
            f for f in os.listdir(folder_path)
            if f.endswith(".txt") and not f.startswith(".")
        ]
        for fname in desc_files:
            fpath = os.path.join(folder_path, fname)
            with open(fpath, "r", encoding="utf-8") as f_in:
                desc_str = f_in.read()

            feats = extract_7_numbers(desc_str)
            data_rows.append(feats + [label])

    columns = [
        "num_nodes",
        "num_edges",
        "avg_degree",
        "num_triangles",
        "global_clustering",
        "max_k_core",
        "num_communities",
        "data_type"
    ]
    df = pd.DataFrame(data_rows, columns=columns)
    return df

###############################################################################
# 3. ANALYSIS & VISUALIZATION
###############################################################################
def analyze_distributions(df, fig_dir="fig"):
    """
    Creates:
      A) One combined set of plots with all data_types together (except 'num_nodes').
      B) One set of plots per folder (data_type) separately.

    Saves everything to `fig_dir`.
    """
    os.makedirs(fig_dir, exist_ok=True)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Exclude 'num_nodes' from the features to plot, per your request
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    all_features = [
        "num_edges", "avg_degree", "num_triangles",
        "global_clustering", "max_k_core", "num_communities"
    ]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3A. COMBINED PLOTS (All data types)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --- Pairplot with different markers & transparency ---
    sns.set_style("whitegrid")
    markers = ["o", "s", "^", "X", "D", "v"]  # Up to 6 distinct shapes
    unique_types = df["data_type"].unique().tolist()
    # If you have more than 6 data types, you'll need more markers or to reuse them

    print("Creating combined pairplot for all data types... (excluding num_nodes)")
    g = sns.pairplot(
        df,
        hue="data_type",
        vars=all_features,
        corner=True,              # Plot only lower triangle
        diag_kind="kde",          # KDE on the diagonal
        plot_kws=dict(alpha=0.6, s=40),
        markers=[markers[i % len(markers)] for i in range(len(unique_types))]
    )
    g.fig.suptitle("Pairwise relationships (combined)", y=1.02)
    plt.tight_layout()
    combined_pairplot_path = os.path.join(fig_dir, "combined_pairplot.png")
    plt.savefig(combined_pairplot_path, dpi=300)
    plt.close()

    # --- PCA scatter plot for all data types ---
    features_df = df[all_features].copy()
    X = features_df.values
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    df["pca1"] = X_pca[:, 0]
    df["pca2"] = X_pca[:, 1]

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=df,
        x="pca1",
        y="pca2",
        hue="data_type",
        style="data_type",
        alpha=0.7,
        s=60,
        markers=[markers[i % len(markers)] for i in range(len(unique_types))]
    )
    plt.title("PCA (2D) of the 6-feature space (combined)")
    plt.tight_layout()
    combined_pca_path = os.path.join(fig_dir, "combined_pca.png")
    plt.savefig(combined_pca_path, dpi=300)
    plt.close()

    print("Combined plots saved.")
    print("Now creating separate plots for each data_type...")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3B. SEPARATE PLOTS (One folder/data_type at a time)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for label in unique_types:
        subset = df[df["data_type"] == label].copy()

        # If there's fewer than 2 samples, pairplot or PCA is not very meaningful
        if len(subset) < 2:
            print(f"Skipping separate plots for '{label}' (need >=2 rows).")
            continue

        # -- Pairplot for this single data_type (no hue) --
        print(f"  Creating pairplot for data_type='{label}' ...")
        g_sub = sns.pairplot(
            subset,
            vars=all_features,
            corner=True,
            diag_kind="kde",
            plot_kws=dict(alpha=0.8, s=45, color="blue"),  # single color
        )
        g_sub.fig.suptitle(f"{label} Only: Pairwise relationships", y=1.02)
        plt.tight_layout()
        pairplot_file = os.path.join(fig_dir, f"{label}_pairplot.png")
        plt.savefig(pairplot_file, dpi=300)
        plt.close()

        # -- PCA for this single data_type --
        X_sub = subset[all_features].values
        pca_sub = PCA(n_components=2)
        X_sub_pca = pca_sub.fit_transform(X_sub)
        subset["pca1"] = X_sub_pca[:, 0]
        subset["pca2"] = X_sub_pca[:, 1]

        plt.figure(figsize=(6, 5))
        plt.scatter(
            subset["pca1"], subset["pca2"],
            alpha=0.7, s=60, c="blue"
        )
        plt.title(f"{label} Only: PCA(2D) of 6D feature space")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        pca_file = os.path.join(fig_dir, f"{label}_pca.png")
        plt.savefig(pca_file, dpi=300)
        plt.close()

    print(f"All separate plots saved in '{fig_dir}'. Done!")

###############################################################################
# 4. PUTTING IT ALL TOGETHER
###############################################################################
if __name__ == "__main__":
    # Example usage
    base_path = "/Users/theo/MVA/Altegrad/Final_project/altegrad_project/"
    folders = {
        "original": os.path.join(base_path, "data", "train", "description"),
        "add_edge": os.path.join(base_path, "data_augmented", "adding_edges", "description"),
        "add_triangle": os.path.join(base_path, "data_augmented", "adding_triangles", "description"),
        # Potentially more, e.g. "add_node": ...
    }

    df_all = collect_description_features(folders)
    print(f"Collected {len(df_all)} total rows across {df_all['data_type'].nunique()} data types.")

    analyze_distributions(df_all, fig_dir="fig")
    print("All plots generated!")
