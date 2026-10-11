# ================================================================
# DBSCAN CLUSTERING
# SKILL MATCH PROJECT - PREPROCESSED DATASET
# ================================================================

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

warnings.filterwarnings("ignore")

# ================================================================
# 1. PATHS
# ================================================================

BASE_DIR = r"C:\Users\chand\Downloads\Skill_Match (2)\Skill_Match"

INPUT_FILE = os.path.join(
    BASE_DIR, "dataset", "reordered_raw.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "output", "DBSCAN_Pre_Processed_dataset_M4_outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# 2. FEATURES
# ================================================================

FEATURES = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]

# ================================================================
# 3. LOAD DATASET
# ================================================================

start_time = time.time()

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}"
    )

data = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully!")
print("Original dataset shape:", data.shape)

missing_columns = [
    col for col in FEATURES if col not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Save selected features
selected_data = data[FEATURES].copy()

selected_data.to_csv(
    os.path.join(OUTPUT_DIR, "selected_features.csv"),
    index=False
)

# ================================================================
# 4. HANDLE MISSING VALUES
# ================================================================

dbscan_data = selected_data.copy()

# Convert selected columns to numeric
for col in FEATURES:
    dbscan_data[col] = pd.to_numeric(
        dbscan_data[col], errors="coerce"
    )

# Replace infinite values with NaN
dbscan_data.replace(
    [np.inf, -np.inf], np.nan, inplace=True
)

# Remove rows containing missing values
dbscan_data.dropna(inplace=True)

if dbscan_data.empty:
    raise ValueError(
        "No valid rows remain after handling missing values."
    )

print("Rows used for clustering:", len(dbscan_data))

# ================================================================
# 5. FEATURE SCALING
# ================================================================

scaler = StandardScaler()

scaled_data = scaler.fit_transform(dbscan_data)

scaled_df = pd.DataFrame(
    scaled_data,
    columns=FEATURES,
    index=dbscan_data.index
)

scaled_df.to_csv(
    os.path.join(OUTPUT_DIR, "standardized_features.csv"),
    index=False
)

# ================================================================
# 6. APPLY DBSCAN
# ================================================================

EPS = 0.80
MIN_SAMPLES = 10

dbscan = DBSCAN(
    eps=EPS,
    min_samples=MIN_SAMPLES
)

labels = dbscan.fit_predict(scaled_data)

clustered_data = data.loc[dbscan_data.index].copy()

clustered_data["DBSCAN_Cluster"] = labels

clustered_data.to_csv(
    os.path.join(OUTPUT_DIR, "DBSCAN_clustered_data.csv"),
    index=False
)

print("\nDBSCAN clustering completed!")

# ================================================================
# 7. CLUSTER DISTRIBUTION
# ================================================================

cluster_counts = pd.Series(labels).value_counts().sort_index()

distribution_df = pd.DataFrame({
    "Cluster": cluster_counts.index,
    "Number_of_Records": cluster_counts.values
})

distribution_df["Cluster_Type"] = distribution_df[
    "Cluster"
].apply(
    lambda x: "Noise" if x == -1 else f"Cluster {x}"
)

distribution_df.to_csv(
    os.path.join(OUTPUT_DIR, "cluster_distribution.csv"),
    index=False
)

print("\nCluster distribution:")
print(distribution_df)

# ================================================================
# 8. CLUSTER SUMMARY
# ================================================================

summary_data = clustered_data.copy()

cluster_summary = summary_data.groupby(
    "DBSCAN_Cluster"
)[FEATURES].mean().round(2)

cluster_summary.to_csv(
    os.path.join(OUTPUT_DIR, "cluster_summary.csv")
)

print("\nCluster summary:")
print(cluster_summary)

# ================================================================
# 9. CLUSTER PERFORMANCE METRICS
# ================================================================

unique_labels = set(labels)

# Exclude noise points when calculating metrics
non_noise_mask = labels != -1
non_noise_labels = labels[non_noise_mask]
non_noise_data = scaled_data[non_noise_mask]

number_of_clusters = len(set(non_noise_labels))

silhouette = np.nan
calinski_harabasz = np.nan
davies_bouldin = np.nan

if (
    number_of_clusters >= 2
    and len(non_noise_labels) > number_of_clusters
):
    silhouette = silhouette_score(
        non_noise_data,
        non_noise_labels
    )

    calinski_harabasz = calinski_harabasz_score(
        non_noise_data,
        non_noise_labels
    )

    davies_bouldin = davies_bouldin_score(
        non_noise_data,
        non_noise_labels
    )

performance_df = pd.DataFrame({
    "Metric": [
        "Total Records Used",
        "Number of Clusters (excluding noise)",
        "Number of Noise Points",
        "Silhouette Score",
        "Calinski-Harabasz Score",
        "Davies-Bouldin Score",
        "Epsilon",
        "Minimum Samples"
    ],
    "Value": [
        len(labels),
        number_of_clusters,
        int(np.sum(labels == -1)),
        silhouette,
        calinski_harabasz,
        davies_bouldin,
        EPS,
        MIN_SAMPLES
    ]
})

performance_df.to_csv(
    os.path.join(OUTPUT_DIR, "performance_metrics.csv"),
    index=False
)

print("\nPerformance metrics:")
print(performance_df)

# ================================================================
# 10. PCA FOR VISUALIZATION
# ================================================================

pca = PCA(n_components=2)

pca_result = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame({
    "PCA1": pca_result[:, 0],
    "PCA2": pca_result[:, 1],
    "DBSCAN_Cluster": labels
})

pca_df.to_csv(
    os.path.join(OUTPUT_DIR, "PCA_coordinates.csv"),
    index=False
)

# ================================================================
# 11. GRAPH 1: AGE VS MONTHLY INCOME
# ================================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    dbscan_data["Age"],
    dbscan_data["MonthlyIncome"],
    c=labels,
    cmap="viridis",
    s=30
)

plt.xlabel("Age")
plt.ylabel("Monthly Income")
plt.title("DBSCAN: Age vs Monthly Income")
plt.colorbar(label="Cluster")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "age_vs_monthly_income.png"),
    dpi=300
)

plt.close()

# ================================================================
# 12. GRAPH 2: AGE VS TOTAL WORKING YEARS
# ================================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    dbscan_data["Age"],
    dbscan_data["TotalWorkingYears"],
    c=labels,
    cmap="viridis",
    s=30
)

plt.xlabel("Age")
plt.ylabel("Total Working Years")
plt.title("DBSCAN: Age vs Total Working Years")
plt.colorbar(label="Cluster")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "age_vs_total_working_years.png"),
    dpi=300
)

plt.close()

# ================================================================
# 13. GRAPH 3: MONTHLY INCOME VS YEARS AT COMPANY
# ================================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    dbscan_data["MonthlyIncome"],
    dbscan_data["YearsAtCompany"],
    c=labels,
    cmap="viridis",
    s=30
)

plt.xlabel("Monthly Income")
plt.ylabel("Years at Company")
plt.title("DBSCAN: Monthly Income vs Years at Company")
plt.colorbar(label="Cluster")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "monthly_income_vs_years_at_company.png"
    ),
    dpi=300
)

plt.close()

# ================================================================
# 14. GRAPH 4: PCA CLUSTER VISUALIZATION
# ================================================================

plt.figure(figsize=(9, 7))

scatter = plt.scatter(
    pca_result[:, 0],
    pca_result[:, 1],
    c=labels,
    cmap="viridis",
    s=30
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("DBSCAN Clusters using PCA")

plt.colorbar(scatter, label="Cluster")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "PCA_DBSCAN_clusters.png"),
    dpi=300
)

plt.close()

# ================================================================
# 15. GRAPH 5: CLUSTER DISTRIBUTION
# ================================================================

plt.figure(figsize=(8, 6))

cluster_counts.plot(kind="bar")

plt.xlabel("Cluster (-1 represents noise)")
plt.ylabel("Number of Records")
plt.title("DBSCAN Cluster Distribution")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "cluster_distribution.png"),
    dpi=300
)

plt.close()

# ================================================================
# 16. SAVE EXECUTION INFORMATION
# ================================================================

execution_time = time.time() - start_time

execution_info = pd.DataFrame({
    "Parameter": [
        "Input File",
        "Output Directory",
        "Original Dataset Rows",
        "Rows Used for Clustering",
        "Number of Features",
        "DBSCAN Epsilon",
        "DBSCAN Minimum Samples",
        "Execution Time (seconds)"
    ],
    "Value": [
        INPUT_FILE,
        OUTPUT_DIR,
        len(data),
        len(dbscan_data),
        len(FEATURES),
        EPS,
        MIN_SAMPLES,
        round(execution_time, 4)
    ]
})

execution_info.to_csv(
    os.path.join(OUTPUT_DIR, "execution_information.csv"),
    index=False
)

# ================================================================
# 17. FINAL RESULTS
# ================================================================

print("\n==========================================")
print("DBSCAN EXECUTION COMPLETED SUCCESSFULLY")
print("==========================================")

print("Original dataset shape:", data.shape)
print("Rows clustered:", len(dbscan_data))
print("Features used:", len(FEATURES))
print("Clusters found:", number_of_clusters)
print("Noise points:", int(np.sum(labels == -1)))
print("Execution time:", round(execution_time, 4), "seconds")
print("Output folder:", OUTPUT_DIR)

print("\nGenerated files:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("-", filename)

print("\nAll outputs saved successfully!")