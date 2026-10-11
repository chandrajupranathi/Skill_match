# ==============================================================
# SKILL MATCH - HIERARCHICAL CLUSTERING
# Agglomerative + Divisive Hierarchical Clustering
#
# IMPORTANT:
# - Original dataset is NOT modified
# - Only a copy/sample is used for clustering
# - All outputs are stored in ONE output folder
# - No TensorFlow required
# ==============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from scipy.cluster.hierarchy import linkage, dendrogram

warnings.filterwarnings("ignore")


# ==============================================================
# 1. PATH SETTINGS
# ==============================================================

# Input: Skill Match Dataset
INPUT_FILE = (
    r"C:\Users\chand\Downloads\Skill_Match (2)\Skill_Match"
    r"\dataset\reordered_raw.csv"
)

# ONE folder for all outputs
OUTPUT_FOLDER = (
    r"C:\Users\chand\Downloads\Skill_Match (2)\Skill_Match"
    r"\output\Hierarchical_Clust_Agglome_Divisive_M4_Outputs"
)

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==============================================================
# 2. PARAMETERS
# ==============================================================

SAMPLE_SIZE = 500
N_CLUSTERS = 3
RANDOM_STATE = 42


# ==============================================================
# 3. LOAD DATASET
# ==============================================================

print("\n==============================================================")
print("SKILL MATCH - HIERARCHICAL CLUSTERING")
print("==============================================================")

print("\nLoading Skill Match dataset...")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nDataset not found:\n{INPUT_FILE}\n"
        "\nPlease check INPUT_FILE path."
    )

# Original dataset is read only
data = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")
print("Original dataset shape:", data.shape)
print("Dataset columns:", data.columns.tolist())


# ==============================================================
# 4. CHECK REQUIRED FEATURES
# ==============================================================

# Preferred Skill Match features
preferred_features = [
    "RequiredSkills",
    "CandidateSkills",
    "SkillScore",
    "SkillMatchScore",
    "MatchScore",
    "Experience",
    "YearsOfExperience",
    "EducationLevel",
    "CGPA",
    "TechnicalSkillsScore",
    "CommunicationScore"
]

print("\nChecking preferred Skill Match features...")

available_features = [
    col for col in preferred_features
    if col in data.columns
    and pd.api.types.is_numeric_dtype(data[col])
]

missing_features = [
    col for col in preferred_features
    if col not in data.columns
]

print("Available features:", available_features)

if missing_features:
    print("Missing features:", missing_features)


# ==============================================================
# 5. SELECT FEATURES
# ==============================================================

if len(available_features) >= 2:

    FEATURES = available_features[:4]

else:

    print("\nWARNING:")
    print("Preferred Skill Match features were not found.")
    print("\nAvailable numeric columns:")

    numeric_columns = data.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    # Avoid using ID columns as clustering features
    excluded_names = {
        "id",
        "candidateid",
        "candidate_id",
        "employeeid",
        "employee_id",
        "userid",
        "user_id"
    }

    numeric_columns = [
        col for col in numeric_columns
        if col.strip().lower().replace(" ", "")
        not in excluded_names
    ]

    print(numeric_columns)

    if len(numeric_columns) < 2:
        raise ValueError(
            "At least two numeric features are required "
            "for hierarchical clustering."
        )

    FEATURES = numeric_columns[:4]

    print("\nUsing these numeric features:")
    print(FEATURES)

print("\nFinal clustering features:")

for feature in FEATURES:
    print(" -", feature)


# ==============================================================
# 6. CREATE A SEPARATE COPY
# ==============================================================

# The original dataset is never changed
working_data = data[FEATURES].copy()


# ==============================================================
# 7. HANDLE MISSING / INFINITE VALUES
# ==============================================================

print("\nChecking missing values...")
print(working_data.isnull().sum())

working_data = working_data.replace(
    [np.inf, -np.inf],
    np.nan
)

# Median imputation on clustering copy only
working_data = working_data.fillna(
    working_data.median(numeric_only=True)
)

# Remaining missing values
working_data = working_data.fillna(0)

print("\nMissing values after cleaning:")
print(working_data.isnull().sum())


# ==============================================================
# 8. SAMPLE DATA
# ==============================================================

if len(working_data) > SAMPLE_SIZE:

    print(f"\nDataset contains {len(working_data)} records.")

    print(
        f"Randomly selecting {SAMPLE_SIZE} records "
        "for hierarchical clustering..."
    )

    sample_data = working_data.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    ).copy()

else:

    print(
        f"\nDataset contains only {len(working_data)} records."
    )

    sample_data = working_data.copy()

if len(sample_data) < 2:
    raise ValueError(
        "At least two records are required for clustering."
    )

print(
    "Records used for hierarchical clustering:",
    len(sample_data)
)

effective_clusters = min(
    N_CLUSTERS,
    len(sample_data)
)


# ==============================================================
# 9. STANDARDIZATION
# ==============================================================

print("\nStandardizing clustering features...")

scaler = StandardScaler()

X = scaler.fit_transform(sample_data)

print("Standardization completed.")


# ==============================================================
# 10. SAVE CLUSTERING INPUT DATA
# ==============================================================

clustering_input = sample_data.copy()

clustering_input.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "hierarchical_clustering_input.csv"
    ),
    index=False
)


# ==============================================================
# 11. AGGLOMERATIVE HIERARCHICAL CLUSTERING
# ==============================================================

print("\n==============================================================")
print("AGGLOMERATIVE HIERARCHICAL CLUSTERING")
print("==============================================================")

print("\nGenerating agglomerative linkage matrix...")

agg_linkage = linkage(
    X,
    method="ward"
)

print("Linkage matrix generated.")


# --------------------------------------------------------------
# Agglomerative Dendrogram
# --------------------------------------------------------------

print("\nCreating Agglomerative Dendrogram...")

plt.figure(figsize=(16, 8))

dendrogram(
    agg_linkage,
    truncate_mode="lastp",
    p=30,
    leaf_rotation=90,
    leaf_font_size=9,
    show_contracted=True
)

plt.title(
    "Agglomerative Hierarchical Clustering Dendrogram",
    fontsize=16
)

plt.xlabel("Cluster / Sample Groups")
plt.ylabel("Ward Distance")
plt.tight_layout()

agg_dendrogram_file = os.path.join(
    OUTPUT_FOLDER,
    "Agglomerative_Dendrogram.png"
)

plt.savefig(
    agg_dendrogram_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", agg_dendrogram_file)


# --------------------------------------------------------------
# Agglomerative Clustering Model
# --------------------------------------------------------------

print("\nPerforming Agglomerative clustering...")

agg_model = AgglomerativeClustering(
    n_clusters=effective_clusters,
    linkage="ward"
)

agg_labels = agg_model.fit_predict(X)

print("Agglomerative clustering completed.")


# ==============================================================
# 12. DIVISIVE HIERARCHICAL CLUSTERING
# ==============================================================

print("\n==============================================================")
print("DIVISIVE HIERARCHICAL CLUSTERING")
print("==============================================================")

print(
    "\nDivisive clustering uses recursive binary K-Means splitting."
)


# --------------------------------------------------------------
# Divisive clustering function
# --------------------------------------------------------------

def divisive_clustering(
    X_data,
    n_clusters=3,
    random_state=42
):
    """
    Starts with one cluster and recursively divides
    the largest cluster into two using K-Means.
    """

    n_samples = len(X_data)

    # Initially all samples belong to cluster 0
    clusters = {
        0: np.arange(n_samples)
    }

    next_cluster_id = 1

    while len(clusters) < n_clusters:

        # Find the largest splittable cluster
        splittable_clusters = [
            (cid, indices)
            for cid, indices in clusters.items()
            if len(indices) >= 2
        ]

        if not splittable_clusters:
            break

        cluster_id, indices = max(
            splittable_clusters,
            key=lambda item: len(item[1])
        )

        cluster_X = X_data[indices]

        # Split cluster into two groups
        kmeans = KMeans(
            n_clusters=2,
            random_state=random_state,
            n_init=10
        )

        split_labels = kmeans.fit_predict(cluster_X)

        group_0 = indices[split_labels == 0]
        group_1 = indices[split_labels == 1]

        if len(group_0) == 0 or len(group_1) == 0:
            break

        # Remove original cluster
        del clusters[cluster_id]

        # Keep first group with original ID
        clusters[cluster_id] = group_0

        # Assign new ID to second group
        clusters[next_cluster_id] = group_1

        next_cluster_id += 1

    # Convert cluster dictionary into labels
    final_labels = np.zeros(
        n_samples,
        dtype=int
    )

    for new_label, indices in enumerate(clusters.values()):
        final_labels[indices] = new_label

    return final_labels


# --------------------------------------------------------------
# Run Divisive Clustering
# --------------------------------------------------------------

div_labels = divisive_clustering(
    X,
    n_clusters=effective_clusters,
    random_state=RANDOM_STATE
)

print("Divisive clustering completed.")

print(
    "Number of divisive clusters:",
    len(np.unique(div_labels))
)


# ==============================================================
# 13. CREATE DIVISIVE DENDROGRAM
# ==============================================================

print("\nCreating Divisive Dendrogram...")

# This centroid linkage is a visualization reference.
# It is not the exact tree generated by the divisive K-Means splits.

div_linkage = linkage(
    X,
    method="centroid"
)

plt.figure(figsize=(16, 8))

dendrogram(
    div_linkage,
    truncate_mode="lastp",
    p=30,
    leaf_rotation=90,
    leaf_font_size=9,
    show_contracted=True
)

plt.title(
    "Divisive Hierarchical Clustering Dendrogram\n"
    "(Centroid-based visualization)",
    fontsize=15
)

plt.xlabel("Cluster / Sample Groups")
plt.ylabel("Centroid Distance")
plt.tight_layout()

div_dendrogram_file = os.path.join(
    OUTPUT_FOLDER,
    "Divisive_Dendrogram.png"
)

plt.savefig(
    div_dendrogram_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", div_dendrogram_file)


# ==============================================================
# 14. EVALUATION FUNCTION
# ==============================================================

def calculate_metrics(X_data, labels):
    """Calculate clustering evaluation metrics."""

    unique_labels = np.unique(labels)

    if (
        len(unique_labels) < 2
        or len(unique_labels) >= len(X_data)
    ):
        return {
            "Silhouette Score": np.nan,
            "Calinski-Harabasz Index": np.nan,
            "Davies-Bouldin Index": np.nan
        }

    return {
        "Silhouette Score": silhouette_score(
            X_data,
            labels
        ),

        "Calinski-Harabasz Index": calinski_harabasz_score(
            X_data,
            labels
        ),

        "Davies-Bouldin Index": davies_bouldin_score(
            X_data,
            labels
        )
    }


# ==============================================================
# 15. EVALUATE AGGLOMERATIVE
# ==============================================================

agg_metrics = calculate_metrics(
    X,
    agg_labels
)


# ==============================================================
# 16. EVALUATE DIVISIVE
# ==============================================================

div_metrics = calculate_metrics(
    X,
    div_labels
)


# ==============================================================
# 17. CREATE RESULTS TABLE
# ==============================================================

results = pd.DataFrame({

    "Method": [
        "Agglomerative",
        "Divisive"
    ],

    "Number_of_Records": [
        len(X),
        len(X)
    ],

    "Number_of_Features": [
        len(FEATURES),
        len(FEATURES)
    ],

    "Number_of_Clusters": [
        len(np.unique(agg_labels)),
        len(np.unique(div_labels))
    ],

    "Silhouette_Score": [
        agg_metrics["Silhouette Score"],
        div_metrics["Silhouette Score"]
    ],

    "Calinski_Harabasz_Index": [
        agg_metrics["Calinski-Harabasz Index"],
        div_metrics["Calinski-Harabasz Index"]
    ],

    "Davies_Bouldin_Index": [
        agg_metrics["Davies-Bouldin Index"],
        div_metrics["Davies-Bouldin Index"]
    ]
})


# ==============================================================
# 18. SAVE RESULTS
# ==============================================================

results_file = os.path.join(
    OUTPUT_FOLDER,
    "Hierarchical_Clustering_Results.csv"
)

results.to_csv(
    results_file,
    index=False
)

print("\nResults saved:")
print(results_file)


# ==============================================================
# 19. SAVE AGGLOMERATIVE CLUSTER ASSIGNMENTS
# ==============================================================

agg_output = sample_data.copy()

agg_output["Agglomerative_Cluster"] = agg_labels

agg_file = os.path.join(
    OUTPUT_FOLDER,
    "Agglomerative_Cluster_Assignments.csv"
)

agg_output.to_csv(
    agg_file,
    index=False
)

print("Saved:", agg_file)


# ==============================================================
# 20. SAVE DIVISIVE CLUSTER ASSIGNMENTS
# ==============================================================

div_output = sample_data.copy()

div_output["Divisive_Cluster"] = div_labels

div_file = os.path.join(
    OUTPUT_FOLDER,
    "Divisive_Cluster_Assignments.csv"
)

div_output.to_csv(
    div_file,
    index=False
)

print("Saved:", div_file)


# ==============================================================
# 21. SAVE COMBINED CLUSTER RESULTS
# ==============================================================

combined_output = sample_data.copy()

combined_output["Agglomerative_Cluster"] = agg_labels
combined_output["Divisive_Cluster"] = div_labels

combined_file = os.path.join(
    OUTPUT_FOLDER,
    "Combined_Hierarchical_Clustering_Results.csv"
)

combined_output.to_csv(
    combined_file,
    index=False
)

print("Saved:", combined_file)


# ==============================================================
# 22. CLUSTER DISTRIBUTION
# ==============================================================

print("\n==============================================================")
print("CLUSTER DISTRIBUTION")
print("==============================================================")

agg_distribution = pd.Series(
    agg_labels
).value_counts().sort_index()

div_distribution = pd.Series(
    div_labels
).value_counts().sort_index()

print("\nAgglomerative Cluster Distribution:")
print(agg_distribution)

print("\nDivisive Cluster Distribution:")
print(div_distribution)


# ==============================================================
# 23. SAVE CLUSTER DISTRIBUTION
# ==============================================================

distribution_rows = []

for cluster, count in agg_distribution.items():

    distribution_rows.append([
        "Agglomerative",
        cluster,
        count
    ])

for cluster, count in div_distribution.items():

    distribution_rows.append([
        "Divisive",
        cluster,
        count
    ])

distribution_df = pd.DataFrame(
    distribution_rows,
    columns=[
        "Method",
        "Cluster",
        "Number_of_Records"
    ]
)

distribution_file = os.path.join(
    OUTPUT_FOLDER,
    "Cluster_Distribution.csv"
)

distribution_df.to_csv(
    distribution_file,
    index=False
)


# ==============================================================
# 24. VISUALIZE CLUSTER DISTRIBUTION
# ==============================================================

plt.figure(figsize=(10, 6))

agg_counts = [
    agg_distribution.get(i, 0)
    for i in range(effective_clusters)
]

div_counts = [
    div_distribution.get(i, 0)
    for i in range(effective_clusters)
]

x = np.arange(effective_clusters)

width = 0.35

plt.bar(
    x - width / 2,
    agg_counts,
    width,
    label="Agglomerative"
)

plt.bar(
    x + width / 2,
    div_counts,
    width,
    label="Divisive"
)

plt.xlabel("Cluster")
plt.ylabel("Number of Records")

plt.title(
    "Cluster Distribution - Skill Match Hierarchical Clustering"
)

plt.xticks(
    x,
    [f"Cluster {i}" for i in range(effective_clusters)]
)

plt.legend()
plt.tight_layout()

distribution_plot = os.path.join(
    OUTPUT_FOLDER,
    "Cluster_Distribution.png"
)

plt.savefig(
    distribution_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==============================================================
# 25. 2D CLUSTER VISUALIZATION
# ==============================================================

# Use first two selected features
if X.shape[1] >= 2:

    plt.figure(figsize=(10, 7))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=agg_labels,
        alpha=0.7
    )

    plt.xlabel(FEATURES[0])
    plt.ylabel(FEATURES[1])

    plt.title(
        "Agglomerative Hierarchical Clustering - Skill Match"
    )

    plt.colorbar(label="Cluster")
    plt.tight_layout()

    agg_scatter = os.path.join(
        OUTPUT_FOLDER,
        "Agglomerative_Cluster_Plot.png"
    )

    plt.savefig(
        agg_scatter,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ----------------------------------------------------------
    # Divisive cluster plot
    # ----------------------------------------------------------

    plt.figure(figsize=(10, 7))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=div_labels,
        alpha=0.7
    )

    plt.xlabel(FEATURES[0])
    plt.ylabel(FEATURES[1])

    plt.title(
        "Divisive Hierarchical Clustering - Skill Match"
    )

    plt.colorbar(label="Cluster")
    plt.tight_layout()

    div_scatter = os.path.join(
        OUTPUT_FOLDER,
        "Divisive_Cluster_Plot.png"
    )

    plt.savefig(
        div_scatter,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==============================================================
# 26. SAVE DATASET INFORMATION
# ==============================================================

info_file = os.path.join(
    OUTPUT_FOLDER,
    "Hierarchical_Clustering_Info.txt"
)

with open(
    info_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "SKILL MATCH - HIERARCHICAL CLUSTERING\n"
    )

    file.write(
        "================================================\n\n"
    )

    file.write("Original Dataset:\n")
    file.write(INPUT_FILE + "\n\n")

    file.write("Original Dataset Shape:\n")
    file.write(str(data.shape) + "\n\n")

    file.write("Features Used:\n")

    for feature in FEATURES:
        file.write(f"- {feature}\n")

    file.write("\nRecords Used:\n")
    file.write(str(len(sample_data)) + "\n")

    file.write("\nNumber of Requested Clusters:\n")
    file.write(str(N_CLUSTERS) + "\n")

    file.write("\nNumber of Effective Clusters:\n")
    file.write(str(effective_clusters) + "\n")

    file.write("\nRandom State:\n")
    file.write(str(RANDOM_STATE) + "\n")

    file.write("\nAgglomerative Metrics:\n")

    for key, value in agg_metrics.items():
        file.write(f"{key}: {value}\n")

    file.write("\nDivisive Metrics:\n")

    for key, value in div_metrics.items():
        file.write(f"{key}: {value}\n")

    file.write("\nOriginal dataset was not modified.\n")


# ==============================================================
# 27. FINAL OUTPUT SUMMARY
# ==============================================================

print("\n==============================================================")
print("SKILL MATCH CLUSTERING COMPLETED SUCCESSFULLY")
print("==============================================================")

print("\nOriginal dataset was NOT modified.")

print("\nFeatures used:")
print(FEATURES)

print(
    "\nNumber of records used:",
    len(sample_data)
)

print(
    "\nNumber of requested clusters:",
    N_CLUSTERS
)

print("\n---------------- RESULTS ----------------")
print(results.to_string(index=False))

print("\n---------------- OUTPUT FOLDER ----------------")
print(OUTPUT_FOLDER)

print("\nGenerated files:")

for filename in sorted(os.listdir(OUTPUT_FOLDER)):
    print(" -", filename)

print("\n==============================================================")
print("DONE")
print("==============================================================")