# ============================================================
# SKILL MATCH - PREPROCESSED / RAW DATASET
# K-MEANS AND K-MEANS++ CLUSTERING
#
# Methods:
#   1. K-Means
#   2. K-Means++
#   3. Elbow Method
#   4. Silhouette Score
#   5. Calinski-Harabasz Score
#   6. Davies-Bouldin Score
#   7. PCA Visualization
#   8. 3-D Clustering Visualization
#   9. Cluster Matching Accuracy (if target exists)
#
# IMPORTANT:
# - Uses the Skill Match dataset
# - Original dataset is NOT modified
# - Suitable numerical features are selected automatically
# - Target and ID columns are excluded from clustering
# - Outputs are stored in separate folders
# ============================================================


import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    accuracy_score
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. INPUT FILE
# ============================================================

BASE_DIR = Path(
    r"C:\Users\chand\Downloads\Skill_Match (2)\Skill_Match"
)

INPUT_FILE = BASE_DIR / "dataset" / "reordered_raw.csv"


# ============================================================
# 2. OUTPUT FOLDER
# ============================================================

OUTPUT_FOLDER = (
    BASE_DIR
    / "output"
    / "K_Means_Elbow_Silhouette_M4_Outputs"
)


# ============================================================
# 3. CREATE SEPARATE OUTPUT FOLDERS
# ============================================================

KMEANS_FOLDER = OUTPUT_FOLDER / "KMeans"

KMEANS_PP_FOLDER = OUTPUT_FOLDER / "KMeansPlusPlus"

ELBOW_FOLDER = OUTPUT_FOLDER / "Elbow"

SILHOUETTE_FOLDER = OUTPUT_FOLDER / "Silhouette"

ACCURACY_FOLDER = OUTPUT_FOLDER / "Accuracy"

for folder in [
    KMEANS_FOLDER,
    KMEANS_PP_FOLDER,
    ELBOW_FOLDER,
    SILHOUETTE_FOLDER,
    ACCURACY_FOLDER
]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# 4. LOAD SKILL MATCH DATASET
# ============================================================

print("=" * 75)
print("K-MEANS AND K-MEANS++ CLUSTERING")
print("SKILL MATCH PROJECT")
print("=" * 75)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "\nSkill Match dataset was not found.\n\n"
        f"Expected location:\n{INPUT_FILE}\n\n"
        "Check the dataset filename and project path."
    )

data = pd.read_csv(INPUT_FILE)

if data.empty:
    raise ValueError("The dataset contains no rows.")

print("\nSkill Match dataset loaded successfully.")
print("Rows    :", data.shape[0])
print("Columns :", data.shape[1])


# ============================================================
# 5. DISPLAY AVAILABLE COLUMNS
# ============================================================

print("\nAvailable columns:")
print("-" * 75)

for i, column in enumerate(data.columns, 1):
    print(i, ".", column)


# ============================================================
# 6. COLUMN-NAME NORMALIZATION
# ============================================================

def normalize_column_name(column):
    return (
        str(column)
        .strip()
        .lower()
        .replace("_", "")
        .replace(" ", "")
        .replace("-", "")
    )


normalized_columns = {
    normalize_column_name(column): column
    for column in data.columns
}


# ============================================================
# 7. IDENTIFY POSSIBLE TARGET COLUMNS
# ============================================================

target_candidates = [
    "SkillMatchStatus",
    "SkillMatchResult",
    "MatchStatus",
    "MatchResult",
    "SkillMatch",
    "PlacementStatus",
    "Placement",
    "Placed",
    "Attrition",
    "Target",
    "Label",
    "Status"
]

target_column = None

for candidate in target_candidates:
    normalized_candidate = normalize_column_name(candidate)

    if normalized_candidate in normalized_columns:
        target_column = normalized_columns[normalized_candidate]
        break


# ============================================================
# 8. IDENTIFY ID COLUMNS
# ============================================================

def is_id_column(column):
    name = normalize_column_name(column)

    return (
        name == "id"
        or name.endswith("id")
        or name in [
            "index",
            "serialnumber",
            "rownumber",
            "recordnumber",
            "employeeid",
            "candidateid",
            "userid"
        ]
    )


id_columns = [
    column
    for column in data.columns
    if is_id_column(column)
]


# ============================================================
# 9. SELECT CLUSTERING FEATURES
# ============================================================

print("\n" + "=" * 75)
print("SELECTING SKILL MATCH FEATURES")
print("=" * 75)

excluded_columns = list(id_columns)

if target_column is not None:
    excluded_columns.append(target_column)

candidate_data = data.drop(
    columns=excluded_columns,
    errors="ignore"
).copy()


# Prefer numeric features related to skill matching.
preferred_feature_names = [
    "SkillScore",
    "SkillMatchScore",
    "MatchScore",
    "TechnicalSkillsScore",
    "TechnicalScore",
    "RequiredSkillsScore",
    "CandidateSkillsScore",
    "Experience",
    "YearsOfExperience",
    "EducationLevel",
    "CGPA",
    "CommunicationScore",
    "AptitudeScore",
    "AptitudeTestScore",
    "AptituteTestScore",
    "ProjectScore",
    "CertificationScore",
    "Internships",
    "Age",
    "MonthlyIncome"
]

preferred_normalized = {
    normalize_column_name(name)
    for name in preferred_feature_names
}

numeric_columns = candidate_data.select_dtypes(
    include=np.number
).columns.tolist()

preferred_numeric_columns = [
    column
    for column in numeric_columns
    if normalize_column_name(column) in preferred_normalized
]


# Use preferred numeric features when at least two exist.
if len(preferred_numeric_columns) >= 2:

    feature_columns = preferred_numeric_columns

else:

    # Otherwise, use all available numeric columns.
    feature_columns = numeric_columns.copy()


# If there are not enough numeric columns, encode suitable
# categorical columns as a fallback.
if len(feature_columns) < 2:

    print(
        "\nFewer than two numeric features were found."
    )

    categorical_columns = candidate_data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Avoid encoding columns that look like unique names or IDs.
    usable_categorical_columns = []

    for column in categorical_columns:

        unique_count = candidate_data[column].nunique(
            dropna=True
        )

        if (
            unique_count >= 2
            and unique_count <= min(
                100,
                max(2, int(len(candidate_data) * 0.5))
            )
        ):
            usable_categorical_columns.append(column)

    if len(feature_columns) == 1:
        fallback_data = candidate_data[
            usable_categorical_columns
        ].copy()

    else:
        fallback_data = candidate_data[
            usable_categorical_columns
        ].copy()

    if not fallback_data.empty:

        encoded_data = pd.get_dummies(
            fallback_data,
            dummy_na=False,
            dtype=float
        )

        for column in encoded_data.columns:
            candidate_data[column] = encoded_data[column]

        feature_columns = (
            feature_columns
            + encoded_data.columns.tolist()
        )

        feature_columns = list(dict.fromkeys(feature_columns))


if len(feature_columns) < 2:
    raise ValueError(
        "\nAt least two usable clustering features are required.\n"
        "Check the dataset columns and their data types."
    )


# ============================================================
# 10. DISPLAY SELECTED FEATURES
# ============================================================

print("\nFeatures selected for clustering:")

for feature in feature_columns:
    print("✓", feature)

print("\nTotal features:", len(feature_columns))

if target_column is not None:
    print("Detected target column:", target_column)
else:
    print("No recognized target column detected.")


# ============================================================
# 11. PREPARE FEATURE DATA
# ============================================================

X = candidate_data[feature_columns].copy()

for column in X.columns:
    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# 12. REMOVE UNUSABLE FEATURES
# ============================================================

# A feature with no valid values cannot be imputed.
all_missing_columns = X.columns[
    X.isna().all()
].tolist()

if all_missing_columns:
    print(
        "\nRemoving columns with no valid values:",
        all_missing_columns
    )

    X = X.drop(columns=all_missing_columns)

if X.shape[1] < 2:
    raise ValueError(
        "Fewer than two usable features remain after cleaning."
    )


# ============================================================
# 13. HANDLE MISSING VALUES
# ============================================================

print("\nMissing values before imputation:")
print(X.isnull().sum())

for column in X.columns:

    median_value = X[column].median()

    if pd.isna(median_value):
        median_value = 0

    X[column] = X[column].fillna(median_value)

print("\nMissing values after imputation:")
print(X.isnull().sum())


# ============================================================
# 14. REMOVE CONSTANT FEATURES
# ============================================================

constant_columns = [
    column
    for column in X.columns
    if X[column].nunique(dropna=False) <= 1
]

if constant_columns:
    print("\nRemoving constant features:", constant_columns)
    X = X.drop(columns=constant_columns)

if X.shape[1] < 2:
    raise ValueError(
        "At least two non-constant features are required."
    )


# Save the cleaned feature data separately.
X.to_csv(
    OUTPUT_FOLDER / "skill_match_clustering_input.csv",
    index=False
)

feature_columns = X.columns.tolist()


# ============================================================
# 15. STANDARDIZATION
# ============================================================

# K-Means uses distances. Standardization prevents features
# with larger numeric ranges from dominating the clustering.

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeature standardization completed.")


# ============================================================
# 16. SET VALID K VALUES
# ============================================================

if len(X) < 3:
    raise ValueError(
        "At least three dataset rows are required for clustering."
    )

MAX_K = min(10, len(X) - 1)

K_VALUES = list(range(2, MAX_K + 1))

print("\nK values tested:", K_VALUES)


# ============================================================
# 17. HELPER FUNCTION FOR SILHOUETTE SCORE
# ============================================================

def calculate_silhouette(X_values, labels):

    # Limit the number of samples for large datasets to reduce
    # computation time. The random sample is reproducible.
    sample_size = min(10000, len(X_values))

    return silhouette_score(
        X_values,
        labels,
        sample_size=sample_size if sample_size < len(X_values) else None,
        random_state=42
    )


# ============================================================
# 18. K-MEANS - ELBOW AND SILHOUETTE ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("K-MEANS ANALYSIS")
print("=" * 75)

kmeans_inertia = []
kmeans_silhouette = []

for k in K_VALUES:

    model = KMeans(
        n_clusters=k,
        init="random",
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    inertia = model.inertia_

    silhouette = calculate_silhouette(
        X_scaled,
        labels
    )

    kmeans_inertia.append(inertia)
    kmeans_silhouette.append(silhouette)

    print(
        f"K = {k:2d} | "
        f"Inertia = {inertia:.4f} | "
        f"Silhouette = {silhouette:.4f}"
    )


# ============================================================
# 19. K-MEANS++ - ELBOW AND SILHOUETTE ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("K-MEANS++ ANALYSIS")
print("=" * 75)

kmeans_pp_inertia = []
kmeans_pp_silhouette = []

for k in K_VALUES:

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    inertia = model.inertia_

    silhouette = calculate_silhouette(
        X_scaled,
        labels
    )

    kmeans_pp_inertia.append(inertia)
    kmeans_pp_silhouette.append(silhouette)

    print(
        f"K = {k:2d} | "
        f"Inertia = {inertia:.4f} | "
        f"Silhouette = {silhouette:.4f}"
    )


# ============================================================
# 20. SELECT BEST K
# ============================================================

# Maximum Silhouette Score selects the best K automatically.
# The Elbow graphs should also be inspected.

best_k_kmeans = K_VALUES[
    int(np.argmax(kmeans_silhouette))
]

best_k_kmeans_pp = K_VALUES[
    int(np.argmax(kmeans_pp_silhouette))
]

print("\n" + "=" * 75)
print("SELECTED NUMBER OF CLUSTERS")
print("=" * 75)

print("K-Means best K   :", best_k_kmeans)
print("K-Means++ best K :", best_k_kmeans_pp)


# ============================================================
# 21. SAVE ELBOW GRAPH - K-MEANS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    K_VALUES,
    kmeans_inertia,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method - Skill Match K-Means")
plt.xticks(K_VALUES)
plt.grid(True)
plt.tight_layout()

plt.savefig(
    ELBOW_FOLDER / "elbow_kmeans.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 22. SAVE ELBOW GRAPH - K-MEANS++
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    K_VALUES,
    kmeans_pp_inertia,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method - Skill Match K-Means++")
plt.xticks(K_VALUES)
plt.grid(True)
plt.tight_layout()

plt.savefig(
    ELBOW_FOLDER / "elbow_kmeans_plus_plus.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 23. SAVE SILHOUETTE GRAPH - K-MEANS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    K_VALUES,
    kmeans_silhouette,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score - Skill Match K-Means")
plt.xticks(K_VALUES)
plt.grid(True)
plt.tight_layout()

plt.savefig(
    SILHOUETTE_FOLDER / "silhouette_kmeans.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 24. SAVE SILHOUETTE GRAPH - K-MEANS++
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    K_VALUES,
    kmeans_pp_silhouette,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score - Skill Match K-Means++")
plt.xticks(K_VALUES)
plt.grid(True)
plt.tight_layout()

plt.savefig(
    SILHOUETTE_FOLDER / "silhouette_kmeans_plus_plus.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 25. FINAL K-MEANS MODEL
# ============================================================

kmeans_model = KMeans(
    n_clusters=best_k_kmeans,
    init="random",
    n_init=10,
    random_state=42
)

kmeans_labels = kmeans_model.fit_predict(X_scaled)


# ============================================================
# 26. FINAL K-MEANS++ MODEL
# ============================================================

kmeans_pp_model = KMeans(
    n_clusters=best_k_kmeans_pp,
    init="k-means++",
    n_init=10,
    random_state=42
)

kmeans_pp_labels = kmeans_pp_model.fit_predict(X_scaled)


# ============================================================
# 27. SAVE K-MEANS CLUSTERED DATA
# ============================================================

kmeans_result = X.copy()

kmeans_result["KMeans_Cluster"] = kmeans_labels + 1

kmeans_result.to_csv(
    KMEANS_FOLDER / "kmeans_clustered_data.csv",
    index=False
)


# ============================================================
# 28. SAVE K-MEANS++ CLUSTERED DATA
# ============================================================

kmeans_pp_result = X.copy()

kmeans_pp_result["KMeansPlusPlus_Cluster"] = (
    kmeans_pp_labels + 1
)

kmeans_pp_result.to_csv(
    KMEANS_PP_FOLDER / "kmeans_plus_plus_clustered_data.csv",
    index=False
)


# ============================================================
# 29. FINAL METRICS - K-MEANS
# ============================================================

km_silhouette = calculate_silhouette(
    X_scaled,
    kmeans_labels
)

km_calinski = calinski_harabasz_score(
    X_scaled,
    kmeans_labels
)

km_davies = davies_bouldin_score(
    X_scaled,
    kmeans_labels
)


# ============================================================
# 30. FINAL METRICS - K-MEANS++
# ============================================================

pp_silhouette = calculate_silhouette(
    X_scaled,
    kmeans_pp_labels
)

pp_calinski = calinski_harabasz_score(
    X_scaled,
    kmeans_pp_labels
)

pp_davies = davies_bouldin_score(
    X_scaled,
    kmeans_pp_labels
)


# ============================================================
# 31. SAVE K-MEANS METRICS
# ============================================================

kmeans_metrics = pd.DataFrame({
    "Method": ["K-Means"],
    "Number_of_Clusters": [best_k_kmeans],
    "Inertia": [kmeans_model.inertia_],
    "Silhouette_Score": [km_silhouette],
    "Calinski_Harabasz_Score": [km_calinski],
    "Davies_Bouldin_Score": [km_davies]
})

kmeans_metrics.to_csv(
    KMEANS_FOLDER / "kmeans_metrics.csv",
    index=False
)


# ============================================================
# 32. SAVE K-MEANS++ METRICS
# ============================================================

kmeans_pp_metrics = pd.DataFrame({
    "Method": ["K-Means++"],
    "Number_of_Clusters": [best_k_kmeans_pp],
    "Inertia": [kmeans_pp_model.inertia_],
    "Silhouette_Score": [pp_silhouette],
    "Calinski_Harabasz_Score": [pp_calinski],
    "Davies_Bouldin_Score": [pp_davies]
})

kmeans_pp_metrics.to_csv(
    KMEANS_PP_FOLDER / "kmeans_plus_plus_metrics.csv",
    index=False
)


# ============================================================
# 33. 3-D K-MEANS CLUSTERING GRAPH
# ============================================================

# Clustering uses all selected features.
# This graph displays up to the first three standardized features.

plot_3d = X_scaled[:, :3]

if plot_3d.shape[1] < 3:
    plot_3d = np.column_stack([
        plot_3d,
        np.zeros(len(plot_3d))
    ])

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    plot_3d[:, 0],
    plot_3d[:, 1],
    plot_3d[:, 2],
    c=kmeans_labels,
    s=20,
    alpha=0.7
)

ax.set_xlabel(feature_columns[0])
ax.set_ylabel(feature_columns[1])

ax.set_zlabel(
    feature_columns[2]
    if len(feature_columns) >= 3
    else "Zero axis"
)

ax.set_title("Skill Match - K-Means Clustering")

fig.colorbar(scatter, ax=ax, label="Cluster")

plt.tight_layout()

plt.savefig(
    KMEANS_FOLDER / "kmeans_3D_clustering.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 34. 3-D K-MEANS++ CLUSTERING GRAPH
# ============================================================

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    plot_3d[:, 0],
    plot_3d[:, 1],
    plot_3d[:, 2],
    c=kmeans_pp_labels,
    s=20,
    alpha=0.7
)

ax.set_xlabel(feature_columns[0])
ax.set_ylabel(feature_columns[1])

ax.set_zlabel(
    feature_columns[2]
    if len(feature_columns) >= 3
    else "Zero axis"
)

ax.set_title("Skill Match - K-Means++ Clustering")

fig.colorbar(scatter, ax=ax, label="Cluster")

plt.tight_layout()

plt.savefig(
    KMEANS_PP_FOLDER / "kmeans_plus_plus_3D_clustering.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 35. PCA 2-D VISUALIZATION
# ============================================================

# PCA is used only to visualize the clustering in two dimensions.
# It does not perform the actual clustering.

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

pca_variance = pca.explained_variance_ratio_

print("\nPCA explained variance:")
print("Component 1:", round(pca_variance[0] * 100, 2), "%")
print("Component 2:", round(pca_variance[1] * 100, 2), "%")


# ============================================================
# 36. K-MEANS PCA GRAPH
# ============================================================

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_labels,
    s=20,
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Skill Match - K-Means PCA Visualization")

plt.colorbar(scatter, label="Cluster")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    KMEANS_FOLDER / "kmeans_PCA_clustering.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 37. K-MEANS++ PCA GRAPH
# ============================================================

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_pp_labels,
    s=20,
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Skill Match - K-Means++ PCA Visualization")

plt.colorbar(scatter, label="Cluster")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    KMEANS_PP_FOLDER / "kmeans_plus_plus_PCA_clustering.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 38. CLUSTER COUNTS
# ============================================================

print("\n" + "=" * 75)
print("K-MEANS CLUSTER COUNTS")
print("=" * 75)

print(
    kmeans_result["KMeans_Cluster"]
    .value_counts()
    .sort_index()
)

print("\n" + "=" * 75)
print("K-MEANS++ CLUSTER COUNTS")
print("=" * 75)

print(
    kmeans_pp_result["KMeansPlusPlus_Cluster"]
    .value_counts()
    .sort_index()
)


# Save cluster counts.
cluster_counts = pd.DataFrame({
    "KMeans_Cluster": (
        kmeans_result["KMeans_Cluster"]
        .value_counts()
        .sort_index()
    ),
    "KMeansPlusPlus_Cluster": (
        kmeans_pp_result["KMeansPlusPlus_Cluster"]
        .value_counts()
        .sort_index()
    )
}).fillna(0).astype(int)

cluster_counts.to_csv(
    OUTPUT_FOLDER / "cluster_distribution.csv"
)


# ============================================================
# 39. CLUSTER MATCHING ACCURACY
# ============================================================

def cluster_matching_accuracy(true_values, cluster_values):

    true_values = pd.Series(
        true_values
    ).reset_index(drop=True)

    cluster_values = pd.Series(
        cluster_values
    ).reset_index(drop=True)

    predicted_values = np.empty(
        len(cluster_values),
        dtype=object
    )

    for cluster in np.unique(cluster_values):

        indexes = np.where(
            cluster_values == cluster
        )[0]

        cluster_targets = true_values.iloc[indexes]

        majority = cluster_targets.mode()

        if len(majority) > 0:
            predicted_values[indexes] = majority.iloc[0]

    return accuracy_score(
        true_values,
        predicted_values
    )


# ============================================================
# 40. CALCULATE ACCURACY IF TARGET EXISTS
# ============================================================

if target_column is not None:

    print("\n" + "=" * 75)
    print("CLUSTER MATCHING ACCURACY")
    print("=" * 75)

    target = data[target_column].reset_index(drop=True)

    valid = target.notna()

    target_valid = target[valid].reset_index(drop=True)

    km_labels_valid = pd.Series(
        kmeans_labels
    )[valid].reset_index(drop=True)

    pp_labels_valid = pd.Series(
        kmeans_pp_labels
    )[valid].reset_index(drop=True)

    km_accuracy = cluster_matching_accuracy(
        target_valid,
        km_labels_valid
    )

    pp_accuracy = cluster_matching_accuracy(
        target_valid,
        pp_labels_valid
    )

    print("Target column:", target_column)

    print(
        f"K-Means   : {km_accuracy * 100:.2f}%"
    )

    print(
        f"K-Means++ : {pp_accuracy * 100:.2f}%"
    )

    accuracy_result = pd.DataFrame({
        "Method": ["K-Means", "K-Means++"],
        "Target_Column": [target_column, target_column],
        "Cluster_Matching_Accuracy": [
            km_accuracy,
            pp_accuracy
        ],
        "Accuracy_Percentage": [
            km_accuracy * 100,
            pp_accuracy * 100
        ]
    })

    accuracy_result.to_csv(
        ACCURACY_FOLDER / "skill_match_cluster_matching_accuracy.csv",
        index=False
    )

else:

    print("\nNo recognized target column was found.")
    print(
        "Cluster matching accuracy was not calculated."
    )
    print(
        "K-Means is unsupervised; its cluster IDs are not "
        "actual match predictions."
    )


# ============================================================
# 41. SAVE COMBINED RESULTS
# ============================================================

combined_result = X.copy()

combined_result["KMeans_Cluster"] = kmeans_labels + 1

combined_result["KMeansPlusPlus_Cluster"] = (
    kmeans_pp_labels + 1
)

if target_column is not None:
    combined_result[target_column] = (
        data[target_column].reset_index(drop=True)
    )

combined_result.to_csv(
    OUTPUT_FOLDER / "combined_kmeans_results.csv",
    index=False
)


# ============================================================
# 42. SAVE FEATURE INFORMATION
# ============================================================

with open(
    OUTPUT_FOLDER / "Skill_Match_KMeans_Info.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("SKILL MATCH - K-MEANS CLUSTERING\n")
    file.write("=" * 60 + "\n\n")

    file.write(f"Input dataset: {INPUT_FILE}\n")
    file.write(f"Rows: {len(X)}\n")
    file.write(f"Features used: {len(feature_columns)}\n\n")

    file.write("Features:\n")

    for feature in feature_columns:
        file.write(f"- {feature}\n")

    file.write("\n")

    file.write(f"K-Means best K: {best_k_kmeans}\n")
    file.write(f"K-Means++ best K: {best_k_kmeans_pp}\n\n")

    file.write(f"K-Means inertia: {kmeans_model.inertia_:.4f}\n")
    file.write(f"K-Means silhouette: {km_silhouette:.4f}\n")
    file.write(f"K-Means Calinski-Harabasz: {km_calinski:.4f}\n")
    file.write(f"K-Means Davies-Bouldin: {km_davies:.4f}\n\n")

    file.write(f"K-Means++ inertia: {kmeans_pp_model.inertia_:.4f}\n")
    file.write(f"K-Means++ silhouette: {pp_silhouette:.4f}\n")
    file.write(f"K-Means++ Calinski-Harabasz: {pp_calinski:.4f}\n")
    file.write(f"K-Means++ Davies-Bouldin: {pp_davies:.4f}\n\n")

    file.write(f"Detected target column: {target_column}\n")
    file.write("Original dataset was not modified.\n")


# ============================================================
# 43. FINAL RESULTS
# ============================================================

print("\n" + "=" * 75)
print("FINAL CLUSTERING RESULTS")
print("=" * 75)

print("\nK-MEANS")
print("-" * 40)

print("Best K:", best_k_kmeans)
print("Inertia:", round(kmeans_model.inertia_, 4))
print("Silhouette:", round(km_silhouette, 4))
print("Calinski-Harabasz:", round(km_calinski, 4))
print("Davies-Bouldin:", round(km_davies, 4))

print("\nK-MEANS++")
print("-" * 40)

print("Best K:", best_k_kmeans_pp)
print("Inertia:", round(kmeans_pp_model.inertia_, 4))
print("Silhouette:", round(pp_silhouette, 4))
print("Calinski-Harabasz:", round(pp_calinski, 4))
print("Davies-Bouldin:", round(pp_davies, 4))


# ============================================================
# 44. OUTPUT LOCATION
# ============================================================

print("\n" + "=" * 75)
print("OUTPUTS SAVED SUCCESSFULLY")
print("=" * 75)

print("\nMain folder:")
print(OUTPUT_FOLDER)

print("\nK-Means folder:")
print(KMEANS_FOLDER)

print("\nK-Means++ folder:")
print(KMEANS_PP_FOLDER)

print("\nElbow folder:")
print(ELBOW_FOLDER)

print("\nSilhouette folder:")
print(SILHOUETTE_FOLDER)

print("\nAccuracy folder:")
print(ACCURACY_FOLDER)

print("\nOriginal dataset was NOT modified.")

print("\nPROGRAM COMPLETED SUCCESSFULLY.")

