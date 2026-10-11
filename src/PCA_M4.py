
# ================================================================
# PRINCIPAL COMPONENT ANALYSIS (PCA)
# SKILL_MATCH PROJECT
# ================================================================
#
# IMPORTANT:
#   1. The original dataset is NEVER modified.
#   2. All preprocessing is performed on copies.
#   3. Only numeric feature columns are used for PCA.
#   4. The target column Attrition is not included in PCA.
#
# INPUT:
#   dataset/reordered_raw.csv
#
# OUTPUT:
#   output/PCA_M4_Outputs/
#
# ================================================================

import os
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ================================================================
# 1. START EXECUTION TIMER
# ================================================================

start_time = time.time()


# ================================================================
# 2. DEFINE PROJECT PATHS
# ================================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_FILE = os.path.join(
    BASE_DIR, "dataset", "reordered_raw.csv"
)

OUTPUT_FOLDER = os.path.join(
    BASE_DIR, "output", "PCA_M4_Outputs"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ================================================================
# 3. CHECK INPUT DATASET
# ================================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}"
    )

print("=" * 70)
print("PRINCIPAL COMPONENT ANALYSIS - SKILL_MATCH")
print("=" * 70)

print("\nInput Dataset:", INPUT_FILE)
print("Output Folder:", OUTPUT_FOLDER)


# ================================================================
# 4. LOAD ORIGINAL DATASET
# ================================================================

df_original = pd.read_csv(INPUT_FILE)

print("\nOriginal Dataset Shape:", df_original.shape)
print("\nFirst Five Rows:")
print(df_original.head())


# ================================================================
# 5. SAVE A COPY OF ORIGINAL DATASET
# ================================================================

original_copy_path = os.path.join(
    OUTPUT_FOLDER, "Original_Dataset_Copy.csv"
)

df_original.copy().to_csv(
    original_copy_path, index=False
)

print("\nOriginal dataset copy saved.")


# ================================================================
# 6. DISPLAY DATASET INFORMATION
# ================================================================

print("\nDataset Information:")
df_original.info()

print("\nMissing Values in Each Column:")
print(df_original.isnull().sum())


# ================================================================
# 7. SELECT NUMERIC FEATURE COLUMNS
# ================================================================

numeric_columns = df_original.select_dtypes(
    include=[np.number]
).columns.tolist()

# Exclude the target column if it is numeric.
target_column = "Attrition"

numeric_columns = [
    column for column in numeric_columns
    if column != target_column
]

if len(numeric_columns) < 2:
    raise ValueError(
        "PCA requires at least two numeric feature columns."
    )

df_numeric = df_original[numeric_columns].copy()

print("\nNumber of Numeric Features:", len(numeric_columns))
print("\nNumeric Feature Columns:")
print(numeric_columns)

print("\nNumeric Dataset Shape:", df_numeric.shape)


# ================================================================
# 8. HANDLE MISSING VALUES
# ================================================================

df_numeric = df_numeric.replace(
    [np.inf, -np.inf], np.nan
)

# Remove columns that contain no valid numeric values.
all_missing_columns = df_numeric.columns[
    df_numeric.isnull().all()
].tolist()

if all_missing_columns:
    print("\nRemoving columns containing only missing values:")
    print(all_missing_columns)

    df_numeric = df_numeric.drop(
        columns=all_missing_columns
    )

    numeric_columns = df_numeric.columns.tolist()

if len(numeric_columns) < 2:
    raise ValueError(
        "At least two usable numeric columns are required."
    )

# Fill missing values using the median.
df_numeric = df_numeric.fillna(
    df_numeric.median()
)

missing_values_path = os.path.join(
    OUTPUT_FOLDER, "Numeric_Data_After_Imputation.csv"
)

df_numeric.to_csv(
    missing_values_path, index=False
)

print("\nMissing values handled using median imputation.")
print("Remaining Missing Values:", df_numeric.isnull().sum().sum())


# ================================================================
# 9. STANDARDIZE NUMERIC FEATURES
# ================================================================

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df_numeric)

df_scaled = pd.DataFrame(
    scaled_data,
    columns=numeric_columns
)

scaled_data_path = os.path.join(
    OUTPUT_FOLDER, "Standardized_Data.csv"
)

df_scaled.to_csv(
    scaled_data_path, index=False
)

print("\nStandardized Dataset Shape:", df_scaled.shape)
print("\nFirst Five Rows of Standardized Data:")
print(df_scaled.head())


# ================================================================
# 10. APPLY PCA WITH ALL COMPONENTS
# ================================================================

pca_all = PCA()

pca_all_data = pca_all.fit_transform(scaled_data)

number_of_components = len(
    pca_all.explained_variance_ratio_
)

print("\nTotal Principal Components:", number_of_components)


# ================================================================
# 11. CALCULATE EXPLAINED VARIANCE
# ================================================================

explained_variance = (
    pca_all.explained_variance_ratio_
)

cumulative_variance = np.cumsum(
    explained_variance
)

variance_table = pd.DataFrame({
    "Principal_Component": [
        f"PC{i + 1}"
        for i in range(number_of_components)
    ],
    "Eigenvalue": pca_all.explained_variance_,
    "Explained_Variance_Ratio": explained_variance,
    "Explained_Variance_Percentage": (
        explained_variance * 100
    ),
    "Cumulative_Variance_Percentage": (
        cumulative_variance * 100
    )
})

variance_path = os.path.join(
    OUTPUT_FOLDER, "Explained_Variance_Table.csv"
)

variance_table.to_csv(
    variance_path, index=False
)

print("\nExplained Variance Table:")
print(variance_table.round(4))


# ================================================================
# 12. SAVE PCA TRANSFORMED DATA WITH ALL COMPONENTS
# ================================================================

component_names = [
    f"PC{i + 1}"
    for i in range(number_of_components)
]

df_pca_all = pd.DataFrame(
    pca_all_data,
    columns=component_names
)

pca_all_path = os.path.join(
    OUTPUT_FOLDER, "PCA_All_Components.csv"
)

df_pca_all.to_csv(
    pca_all_path, index=False
)

print("\nPCA Data with All Components Saved.")


# ================================================================
# 13. CALCULATE PCA LOADINGS
# ================================================================

loadings = pca_all.components_.T

df_loadings = pd.DataFrame(
    loadings,
    index=numeric_columns,
    columns=component_names
)

df_loadings.index.name = "Original_Feature"

loadings_path = os.path.join(
    OUTPUT_FOLDER, "PCA_Loadings.csv"
)

df_loadings.to_csv(loadings_path)

print("\nPCA Loadings:")
print(df_loadings.round(4))


# ================================================================
# 14. FIND COMPONENTS REQUIRED FOR VARIANCE THRESHOLDS
# ================================================================

variance_thresholds = [0.80, 0.90, 0.95]

component_counts = {}

for threshold in variance_thresholds:
    required_components = (
        np.argmax(cumulative_variance >= threshold) + 1
    )

    component_counts[threshold] = required_components

print("\nComponents Required for Variance Thresholds:")

for threshold, count in component_counts.items():
    print(
        f"{threshold * 100:.0f}% Variance: "
        f"{count} Components"
    )


# ================================================================
# 15. SAVE COMPONENT COUNT SUMMARY
# ================================================================

component_summary = pd.DataFrame({
    "Variance_Threshold_Percentage": [
        int(threshold * 100)
        for threshold in component_counts
    ],
    "Required_Components": list(
        component_counts.values()
    )
})

component_summary_path = os.path.join(
    OUTPUT_FOLDER, "Component_Count_Summary.csv"
)

component_summary.to_csv(
    component_summary_path, index=False
)

print("\nComponent Count Summary:")
print(component_summary)


# ================================================================
# 16. APPLY PCA WITH 95% VARIANCE RETENTION
# ================================================================

n_components_95 = component_counts[0.95]

pca_95 = PCA(
    n_components=n_components_95
)

pca_95_data = pca_95.fit_transform(
    scaled_data
)

pc95_columns = [
    f"PC{i + 1}"
    for i in range(n_components_95)
]

df_pca_95 = pd.DataFrame(
    pca_95_data,
    columns=pc95_columns
)

pca_95_path = os.path.join(
    OUTPUT_FOLDER, "PCA_95_Percent_Variance.csv"
)

df_pca_95.to_csv(
    pca_95_path, index=False
)

actual_variance_95 = (
    pca_95.explained_variance_ratio_.sum()
)

print("\nPCA with 95% Variance Retention:")
print("Selected Components:", n_components_95)
print(
    "Actual Variance Retained:",
    round(actual_variance_95 * 100, 2), "%"
)
print("Reduced Dataset Shape:", df_pca_95.shape)


# ================================================================
# 17. CREATE PCA SUMMARY RESULTS
# ================================================================

summary_results = pd.DataFrame({
    "Metric": [
        "Original Number of Rows",
        "Original Number of Numeric Features",
        "Number of PCA Components",
        "Components for 80% Variance",
        "Components for 90% Variance",
        "Components for 95% Variance",
        "Selected Components for 95% PCA",
        "Actual Variance Retained by 95% PCA"
    ],
    "Value": [
        len(df_original),
        len(numeric_columns),
        number_of_components,
        component_counts[0.80],
        component_counts[0.90],
        component_counts[0.95],
        n_components_95,
        round(actual_variance_95 * 100, 4)
    ]
})

summary_path = os.path.join(
    OUTPUT_FOLDER, "PCA_Summary_Results.csv"
)

summary_results.to_csv(
    summary_path, index=False
)

print("\nPCA Summary Results:")
print(summary_results)


# ================================================================
# 18. PLOT SCREE PLOT
# ================================================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, number_of_components + 1),
    pca_all.explained_variance_,
    marker="o"
)

plt.title("PCA Scree Plot - Skill_Match")
plt.xlabel("Principal Component")
plt.ylabel("Eigenvalue")
plt.xticks(
    range(1, number_of_components + 1)
)
plt.grid(True)
plt.tight_layout()

scree_path = os.path.join(
    OUTPUT_FOLDER, "PCA_Scree_Plot.png"
)

plt.savefig(scree_path, dpi=300)
plt.close()

print("\nScree plot saved.")


# ================================================================
# 19. PLOT INDIVIDUAL EXPLAINED VARIANCE
# ================================================================

plt.figure(figsize=(10, 6))

plt.bar(
    range(1, number_of_components + 1),
    explained_variance * 100
)

plt.title("Explained Variance by Principal Component")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance (%)")
plt.xticks(
    range(1, number_of_components + 1)
)
plt.grid(axis="y")
plt.tight_layout()

individual_variance_path = os.path.join(
    OUTPUT_FOLDER, "Individual_Explained_Variance.png"
)

plt.savefig(individual_variance_path, dpi=300)
plt.close()

print("\nIndividual explained variance plot saved.")


# ================================================================
# 20. PLOT CUMULATIVE EXPLAINED VARIANCE
# ================================================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, number_of_components + 1),
    cumulative_variance * 100,
    marker="o"
)

plt.axhline(
    y=80, linestyle="--", label="80% Variance"
)

plt.axhline(
    y=90, linestyle="--", label="90% Variance"
)

plt.axhline(
    y=95, linestyle="--", label="95% Variance"
)

plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Variance (%)")
plt.xticks(
    range(1, number_of_components + 1)
)
plt.legend()
plt.grid(True)
plt.tight_layout()

cumulative_path = os.path.join(
    OUTPUT_FOLDER, "Cumulative_Explained_Variance.png"
)

plt.savefig(cumulative_path, dpi=300)
plt.close()

print("\nCumulative variance plot saved.")


# ================================================================
# 21. VISUALIZE DATA USING FIRST TWO COMPONENTS
# ================================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    df_pca_all["PC1"],
    df_pca_all["PC2"],
    alpha=0.6
)

plt.title("PCA Visualization - First Two Components")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()

pca_2d_path = os.path.join(
    OUTPUT_FOLDER, "PCA_2D_Visualization.png"
)

plt.savefig(pca_2d_path, dpi=300)
plt.close()

print("\n2D PCA visualization saved.")


# ================================================================
# 22. PLOT FEATURE CONTRIBUTIONS TO FIRST COMPONENT
# ================================================================

pc1_loadings = df_loadings["PC1"]

top_pc1_features = (
    pc1_loadings.abs()
    .sort_values(ascending=False)
    .head(15)
    .index
)

plt.figure(figsize=(12, 7))

plt.barh(
    top_pc1_features,
    pc1_loadings.loc[top_pc1_features]
)

plt.title("Top Feature Loadings for PC1")
plt.xlabel("Loading Value")
plt.ylabel("Original Numeric Feature")
plt.gca().invert_yaxis()
plt.tight_layout()

pc1_path = os.path.join(
    OUTPUT_FOLDER, "PC1_Top_Feature_Loadings.png"
)

plt.savefig(pc1_path, dpi=300)
plt.close()

print("\nPC1 feature loading plot saved.")


# ================================================================
# 23. SAVE EXECUTION INFORMATION
# ================================================================

execution_time = time.time() - start_time

execution_info = pd.DataFrame({
    "Parameter": [
        "Input Dataset",
        "Output Folder",
        "Original Rows",
        "Original Columns",
        "Numeric Features Used",
        "Components Generated",
        "Components for 95% Variance",
        "Execution Time (Seconds)"
    ],
    "Value": [
        INPUT_FILE,
        OUTPUT_FOLDER,
        df_original.shape[0],
        df_original.shape[1],
        len(numeric_columns),
        number_of_components,
        n_components_95,
        round(execution_time, 4)
    ]
})

execution_info_path = os.path.join(
    OUTPUT_FOLDER, "Execution_Information.csv"
)

execution_info.to_csv(
    execution_info_path, index=False
)

print("\nExecution Information:")
print(execution_info)


# ================================================================
# 24. DISPLAY ALL OUTPUT FILES
# ================================================================

print("\n" + "=" * 70)
print("GENERATED OUTPUT FILES")
print("=" * 70)

for filename in sorted(os.listdir(OUTPUT_FOLDER)):
    file_path = os.path.join(OUTPUT_FOLDER, filename)

    if os.path.isfile(file_path):
        print(filename)


# ================================================================
# 25. FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("PCA EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nOriginal Dataset Shape:", df_original.shape)
print("Numeric Features Used:", len(numeric_columns))
print("Total PCA Components:", number_of_components)
print("Components for 95% Variance:", n_components_95)
print(
    "Variance Retained:",
    round(actual_variance_95 * 100, 2), "%"
)
print("Execution Time:", round(execution_time, 2), "seconds")
print("Output Folder:", OUTPUT_FOLDER)

print("\nIMPORTANT:")
print("The original dataset was not modified.")
print("PCA was performed only on numeric features.")
print("Attrition was excluded from PCA.")

print("\nAll PCA outputs are saved successfully!")
