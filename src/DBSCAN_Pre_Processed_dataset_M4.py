
# ============================================================
# DBSCAN CLUSTERING ON SKILL MATCH DATASET
# Original Dataset is NOT Modified
# All Outputs and Performance Results -> ONE Folder
#
# Does NOT use:
#   silhouette_score
#   calinski_harabasz_score
#   davies_bouldin_score
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# ============================================================
# 1. INPUT AND OUTPUT PATH
# ============================================================

BASE_DIR = (
    r"C:\Users\chand\Downloads\Skill_Match (2)\Skill_Match"
)

INPUT_FILE = os.path.join(
    BASE_DIR, "dataset", "reordered_raw.csv"
)

OUTPUT_FOLDER = os.path.join(
    BASE_DIR, "output",
    "DBSCAN_Pre_Processed_dataset_M4_outputs"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 75)
print("DBSCAN CLUSTERING - SKILL MATCH DATASET")
print("=" * 75)

if not os.path.exists(INPUT_FILE):
    print("\nERROR: Input file not found!")
    print("Check the following path:")
    print(INPUT_FILE)
    raise SystemExit

# Read-only loading: the original file is not modified.
data = pd.read_csv(INPUT_FILE)

print("\nOriginal Dataset Shape:")
print(data.shape)

print("\nAvailable Columns:")
print(list(data.columns))


# ============================================================
# 3. SELECT REQUIRED NUMERIC FEATURES
# ============================================================

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

missing_columns = [
    column for column in FEATURES
    if column not in data.columns
]

if missing_columns:
    print("\nERROR: Required columns are missing:")
    print(missing_columns)
    print("\nPlease check the dataset column names.")
    raise SystemExit

# Create a separate copy of the selected features.
dbscan_data = data[FEATURES].copy()


# ============================================================
# 4. CONVERT FEATURES TO NUMERIC
# ============================================================

for column in FEATURES:
    dbscan_data[column] = pd.to_numeric(
        dbscan_data[column],
        errors="coerce"
    )


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(dbscan_data.isnull().sum())


# ============================================================
# 6. REMOVE INVALID ROWS FROM THE COPY ONLY
# ============================================================

before_cleaning = len(dbscan_data)

dbscan_data.replace(
    [np.inf, -np.inf], np.nan, inplace=True
)

# Keep original row indexes so labels match the correct records.
dbscan_data = dbscan_data.dropna()

after_cleaning = len(dbscan_data)
removed_rows = before_cleaning - after_cleaning

print("\nRows before cleaning:", before_cleaning)
print("Rows after cleaning :", after_cleaning)
print("Rows removed        :", removed_rows)

if len(dbscan_data) < 2:
    raise ValueError(
        "At least two valid rows are required for DBSCAN and PCA."
    )


# ============================================================
# 7. SAVE SELECTED FEATURES
# ============================================================

selected_features_file = os.path.join(
    OUTPUT_FOLDER,
    "DBSCAN_selected_features.csv"
)

dbscan_data.to_csv(
    selected_features_file,
    index=False
)


# ============================================================
# 8. STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    dbscan_data[FEATURES]
)


# ============================================================
# 9. SAVE STANDARDIZED DATA
# ============================================================

standardized_data = pd.DataFrame(
    X_scaled,
    columns=FEATURES
)

standardized_file = os.path.join(
    OUTPUT_FOLDER,
    "DBSCAN_standardized_features.csv"
)

standardized_data.to_csv(
    standardized_file,
    index=False
)


# ============================================================
# 10. DBSCAN PARAMETERS
# ============================================================

EPS = 0.80
MIN_SAMPLES = 10

print("\nDBSCAN Parameters")
print("-" * 40)
print("eps        :", EPS)
print("min_samples:", MIN_SAMPLES)


# ============================================================
# 11. APPLY DBSCAN
# ============================================================

dbscan_model = DBSCAN(
    eps=EPS,
    min_samples=MIN_SAMPLES
)

cluster_labels = dbscan_model.fit_predict(X_scaled)


# ============================================================
# 12. CREATE CLUSTERED DATASET
# ============================================================

# Retain all original columns for the valid records.
clustered_data = data.loc[dbscan_data.index].copy()

clustered_data["DBSCAN_Cluster"] = cluster_labels


# ============================================================
# 13. CLUSTER INFORMATION
# ============================================================

unique_labels = sorted(np.unique(cluster_labels))

actual_clusters = [
    label for label in unique_labels
    if label != -1
]

number_of_clusters = len(actual_clusters)

noise_count = int(np.sum(cluster_labels == -1))

clustered_count = len(cluster_labels) - noise_count
total_records = len(cluster_labels)

if total_records > 0:
    noise_percentage = noise_count / total_records * 100
    clustered_percentage = clustered_count / total_records * 100
else:
    noise_percentage = 0
    clustered_percentage = 0

print("\n" + "=" * 75)
print("DBSCAN RESULTS")
print("=" * 75)

print("\nTotal Records       :", total_records)
print("Number of Clusters  :", number_of_clusters)
print("Noise Points        :", noise_count)
print("Clustered Points    :", clustered_count)
print("Noise Percentage    :", round(noise_percentage, 2), "%")
print("Clustered Percentage:", round(clustered_percentage, 2), "%")


# ============================================================
# 14. SAVE CLUSTERED DATA
# ============================================================

clustered_file = os.path.join(
    OUTPUT_FOLDER,
    "DBSCAN_clustered_data.csv"
)

clustered_data.to_csv(
    clustered_file,
    index=False
)


# ============================================================
# 15. CLUSTER DISTRIBUTION
# ============================================================

cluster_counts = (
    clustered_data["DBSCAN_Cluster"]
    .value_counts()
    .sort_index()
)

print("\nCluster Distribution:")
print(cluster_counts)

cluster_distribution = pd.DataFrame({
    "Cluster": cluster_counts.index,
    "Number_of_Records": cluster_counts.values
})

cluster_distribution_file = os.path.join(
    OUTPUT_FOLDER,
    "DBSCAN_cluster_distribution.csv"
)

cluster_distribution.to_csv(
    cluster_distribution_file,
    index=False
)


# ============================================================
# 16. CLUSTER SUMMARY
# ============================================================

summary_records = []

for cluster_id in unique_labels:

    cluster_subset = clustered_data[
        clustered_data["DBSCAN_Cluster"] == cluster_id
    ]

    if cluster_id == -1:
        cluster_name = "Noise"
    else:
        cluster_name = "Cluster_" + str(cluster_id)

    record = {
        "Cluster": cluster_name,
        "Cluster_ID": int(cluster_id),
        "Number_of_Records": len(cluster_subset)
    }

    for feature in FEATURES:
        record["Average_" + feature] = round(
            cluster_subset[feature].mean(), 3
        )

    summary_records.append(record)

cluster_summary = pd.DataFrame(summary_records)

cluster_summary_file = os.path.join(
    OUTPUT_FOLDER,
    "DBSCAN_cluster_summary.csv"
)

cluster_summary.to_csv(
    cluster_summary_file,
    index=False
)


# ============================================================
# 17. PERFORMANCE RESULTS
# ============================================================

performance_results = pd.DataFrame({
    "Metric": [
        "Algorithm",
        "Input Dataset",
        "Original Dataset Rows",
        "Rows Used",
        "Rows Removed",
        "Number of Features",
        "Features Used",
        "Scaling Method",
        "DBSCAN eps",
        "DBSCAN min_samples",
        "Number of Clusters",
        "Noise Points",
        "Clustered Points",
        "Noise Percentage",
        "Clustered Percentage"
    ],
    "Value": [
        "DBSCAN",
        os.path.basename(INPUT_FILE),
        before_cleaning,
        after_cleaning,
        removed_rows,
        len(FEATURES),
        ", ".join(FEATURES),
        "StandardScaler",
        EPS,
        MIN_SAMPLES,
        number_of_clusters,
        noise_count,
        clustered_count,
        round(noise_percentage, 2),
        round(clustered_percentage, 2)
    ]
})

performance_file = os.path.join(
    OUTPUT_FOLDER,
    "DBSCAN_performance_results.csv"
)

performance_results.to_csv(
    performance_file,
    index=False
)


# ============================================================
# 18. PCA FOR 2-D VISUALIZATION
# ============================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

pca_data = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "DBSCAN_Cluster": cluster_labels
})

pca_file = os.path.join(
    OUTPUT_FOLDER,
    "DBSCAN_PCA_coordinates.csv"
)

pca_data.to_csv(
    pca_file,
    index=False
)


# ============================================================
# 19. GRAPH 1 - AGE VS MONTHLY INCOME
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    dbscan_data["Age"],
    dbscan_data["MonthlyIncome"],
    c=cluster_labels,
    s=25,
    alpha=0.7
)

plt.xlabel("Age")
plt.ylabel("Monthly Income")
plt.title("DBSCAN Clustering: Age vs Monthly Income")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "01_DBSCAN_Age_vs_MonthlyIncome.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 20. GRAPH 2 - AGE VS TOTAL WORKING YEARS
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    dbscan_data["Age"],
    dbscan_data["TotalWorkingYears"],
    c=cluster_labels,
    s=25,
    alpha=0.7
)

plt.xlabel("Age")
plt.ylabel("Total Working Years")
plt.title("DBSCAN Clustering: Age vs Total Working Years")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "02_DBSCAN_Age_vs_TotalWorkingYears.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 21. GRAPH 3 - MONTHLY INCOME VS YEARS AT COMPANY
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    dbscan_data["MonthlyIncome"],
    dbscan_data["YearsAtCompany"],
    c=cluster_labels,
    s=25,
    alpha=0.7
)

plt.xlabel("Monthly Income")
plt.ylabel("Years at Company")
plt.title("DBSCAN Clustering: Monthly Income vs Years at Company")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "03_DBSCAN_MonthlyIncome_vs_YearsAtCompany.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 22. GRAPH 4 - PCA CLUSTER VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_labels,
    s=25,
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("DBSCAN Cluster Visualization using PCA")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "04_DBSCAN_PCA_Clusters.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 23. GRAPH 5 - CLUSTER DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    cluster_distribution["Cluster"].astype(str),
    cluster_distribution["Number_of_Records"]
)

plt.xlabel("DBSCAN Cluster (-1 represents noise)")
plt.ylabel("Number of Records")
plt.title("DBSCAN Cluster Distribution")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "05_DBSCAN_Cluster_Distribution.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 24. SAVE EXECUTION INFORMATION
# ============================================================

execution_information = pd.DataFrame({
    "Item": [
        "Input Dataset",
        "Output Folder",
        "Original Dataset Modified",
        "Features Used",
        "Preprocessing",
        "Scaling",
        "Algorithm",
        "eps",
        "min_samples"
    ],
    "Details": [
        INPUT_FILE,
        OUTPUT_FOLDER,
        "NO",
        ", ".join(FEATURES),
        "Numeric conversion and removal of invalid rows",
        "StandardScaler",
        "DBSCAN",
        EPS,
        MIN_SAMPLES
    ]
})

execution_file = os.path.join(
    OUTPUT_FOLDER,
    "execution_information.csv"
)

execution_information.to_csv(
    execution_file,
    index=False
)


# ============================================================
# 25. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("ALL OUTPUTS GENERATED SUCCESSFULLY")
print("=" * 75)

print("\nOutput Folder:")
print(OUTPUT_FOLDER)

print("\nFiles Generated:")

for filename in sorted(os.listdir(OUTPUT_FOLDER)):
    print("   ", filename)

print("\n" + "=" * 75)
print("DBSCAN ANALYSIS COMPLETED")
print("Original dataset was NOT modified.")
print("=" * 75)
