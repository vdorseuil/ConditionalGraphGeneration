import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

###############################################################################
# 1. PARSING THE 7 NUMERICAL VALUES
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
        ii= 0
        desc_files = [
            f for f in os.listdir(folder_path)
            if f.endswith(".txt") and not f.startswith(".")
        ]
        for ijk in range(8000):
            fname = 'graph_'+str(ijk) +'.txt'
            fpath = os.path.join(folder_path, fname)
            with open(fpath, "r", encoding="utf-8") as f_in:
                desc_str = f_in.read()
                    

            feats = extract_7_numbers(desc_str)
            if ii<3 : 
                print(feats)
                ii +=1
            if ii ==3 : 
                print()
                print()
                ii+=1
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
        plot_kws=dict(alpha=0.3, s=20),
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
        alpha=0.3,
        s=20,
        markers=[markers[i % len(markers)] for i in range(len(unique_types))]
    )
    plt.title("PCA (2D) of the 6-feature space (combined)")
    plt.tight_layout()
    combined_pca_path = os.path.join(fig_dir, "combined_pca.png")
    plt.savefig(combined_pca_path, dpi=300)
    plt.close()

    print("Combined plots saved.")


import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. PARSING THE 7 NUMERICAL VALUES
def extract_7_numbers(description_str: str) -> list[float]:
    nums = re.findall(r"\d+\.\d+|\d+", description_str)
    values = [float(x) for x in nums[:7]]
    return values

# 2. GATHER DATA FROM MULTIPLE FOLDERS
def collect_description_features(folders_dict):
    """
    Reads .txt files from each folder in `folders_dict` and extracts the 7 metrics.
    Ensures that numerical columns are properly converted to float.
    """
    data_rows = []
    for label, folder_path in folders_dict.items():
        # List and sort files numerically by extracting numeric part of filenames
        desc_files = [
            f for f in os.listdir(folder_path)
            if f.endswith(".txt") and not f.startswith(".")
        ]
        desc_files.sort(key=lambda x: int(re.search(r"\d+", x).group()))

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

    # Ensure numerical columns are float
    numeric_columns = columns[:-1]  # Exclude 'data_type'
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    return df

# NEW: Create a DataFrame of differences
def compute_differences(df_all):
    """
    Computes df_diff by copying df_all and calculating row-wise differences for each
    feature across the original and augmented data types.
    """
    df_diff = df_all.copy()
    num_rows = 8000  # Assuming each data type has 8000 rows

    for j in range(1, 4):  # Corresponding to the 3 augmented types
        for i in range(num_rows):
            df_diff.iloc[num_rows * j + i, :-1] = (
                df_all.iloc[num_rows * j + i, :-1] - df_all.iloc[i, :-1]
            )
            if i < 3 : 
                print(df_all.iloc[num_rows * j + i, :-1] )
                print(df_all.iloc[i, :-1])
                

    # Remove rows corresponding to 'original'
    df_diff = df_diff[df_diff["data_type"] != "original"].reset_index(drop=True)
    return df_diff

# 3. ANALYSIS & VISUALIZATION
def analyze_differences(df_differences, fig_dir="fig_diff"):
    os.makedirs(fig_dir, exist_ok=True)

    features = [
        "num_nodes", "num_edges", "avg_degree", "num_triangles",
        "global_clustering", "max_k_core", "num_communities"
    ]

    # Pairplot
    sns.set_style("whitegrid")
    sns.pairplot(
        df_differences,
        vars=features,
        hue="data_type",
        corner=True,
        diag_kind="kde",
        plot_kws=dict(alpha=0.3, s=20)
    )
    plt.suptitle("Pairwise relationships (differences)", y=1.02)
    plt.tight_layout()
    pairplot_path = os.path.join(fig_dir, "differences_pairplot.png")
    plt.savefig(pairplot_path, dpi=300)
    plt.close()

    # PCA
    X = df_differences[features].values
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    df_differences["pca1"] = X_pca[:, 0]
    df_differences["pca2"] = X_pca[:, 1]

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        x="pca1", y="pca2", data=df_differences, hue="data_type", alpha=0.3, s=20
    )
    plt.title("PCA (2D) of the differences")
    plt.tight_layout()
    pca_path = os.path.join(fig_dir, "differences_pca.png")
    plt.savefig(pca_path, dpi=300)
    plt.close()

    print("Difference plots saved.")

###############################################################################
# 4. PUTTING IT ALL TOGETHER
###############################################################################
if __name__ == "__main__":
    base_path = "/"
    folders = {
        "original": os.path.join(base_path, "data", "train", "description"),
        "add_edge": os.path.join(base_path, "data_augmented", "adding_edges", "description"),
        "add_triangle": os.path.join(base_path, "data_augmented", "adding_triangles", "description"),
         "add_nodes": os.path.join(base_path, "data_augmented", "adding_nodes", "description")
    }

    df_all = collect_description_features(folders)
    print(f"Collected {len(df_all)} total rows across {df_all['data_type'].nunique()} data types.")

    print("All plots generated!")
   
    df_diff = compute_differences(df_all)
    print(f"Computed differences with shape: {df_diff.shape}")

    analyze_differences(df_diff, fig_dir="fig_diff")
    print("Difference analysis completed!")
    print(df_diff.head(10))
  
