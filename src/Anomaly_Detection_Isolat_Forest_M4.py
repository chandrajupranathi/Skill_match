
# ============================================================
# ISOLATION FOREST ANOMALY DETECTION
# SKILL MATCH - EMPLOYEE ATTRITION DATASET
#
# IMPORTANT:
# 1. Original dataset is NOT modified.
# 2. All outputs are stored in ONE folder.
# 3. Only Isolation Forest is used for anomaly detection.
# 4. Attrition is excluded from detection features so that
#    the target label does not influence anomaly detection.
# ============================================================

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT, INPUT AND OUTPUT PATHS
# ============================================================

PROJECT_PATH = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_CANDIDATES = [
    os.path.join(
        PROJECT_PATH,
        "dataset",
        "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    ),
    os.path.join(
        PROJECT_PATH,
        "dataset",
        "employee_attrition.csv"
    ),
    os.path.join(
        PROJECT_PATH,
        "static",
        "data",
        "employee_attrition.csv"
    )
]

INPUT_FILE = next(
    (
        path for path in DATASET_CANDIDATES
        if os.path.isfile(path)
    ),
    None
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_PATH,
    "output",
    "Anomaly_Detection_Isolation_Forest_M4_Outputs"
)

# Create ONE output folder.
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. PROGRAM HEADER AND LOAD DATASET
# ============================================================

print("=" * 70)
print("ISOLATION FOREST ANOMALY DETECTION")
print("SKILL MATCH - EMPLOYEE ATTRITION DATASET")
print("=" * 70)

if INPUT_FILE is None:
    print("\nDataset not found. Checked these locations:")

    for path in DATASET_CANDIDATES:
        print(" -", path)

    raise FileNotFoundError(
        "Place the raw employee attrition CSV in the dataset "
        "folder or update DATASET_CANDIDATES."
    )

print("\nInput dataset:", INPUT_FILE)

data = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")
print("Original dataset shape:", data.shape)

if data.empty:
    raise ValueError("The dataset is empty.")

if data.shape[1] < 2:
    raise ValueError(
        "Dataset must contain at least two columns."
    )


# ============================================================
# 3. CREATE A COPY
# ============================================================

# All processing happens on copies.
# The original CSV is never overwritten.

working_data = data.copy()

working_data.columns = (
    working_data.columns.astype(str).str.strip()
)

print("\nDataset columns:")
print(list(working_data.columns))


# ============================================================
# 4. SELECT NUMERIC FEATURES
# ============================================================

# Attrition is a prediction target, not an anomaly feature.
# Exclude it if it exists in the dataset.

target_columns = [
    column for column in working_data.columns
    if column.strip().lower() == "attrition"
]

feature_data = working_data.drop(
    columns=target_columns,
    errors="ignore"
).copy()

numeric_data = feature_data.select_dtypes(
    include=[np.number]
).copy()

print("\nNumber of numeric features:", numeric_data.shape[1])

print("\nNumeric features used:")

for column in numeric_data.columns:
    print(" -", column)

if numeric_data.shape[1] == 0:
    raise ValueError(
        "No numeric employee features found. "
        "Check the input dataset."
    )


# ============================================================
# 5. HANDLE INFINITE VALUES
# ============================================================

numeric_data = numeric_data.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

missing_before = int(
    numeric_data.isnull().sum().sum()
)

print("\nMissing numeric values before processing:",
      missing_before)

# Replace missing values with the column median.
numeric_data = numeric_data.fillna(
    numeric_data.median()
)

# Entirely missing numeric columns may still contain NaN.
numeric_data = numeric_data.fillna(0)

missing_after = int(
    numeric_data.isnull().sum().sum()
)

print("Missing numeric values after processing:",
      missing_after)


# ============================================================
# 7. REMOVE CONSTANT FEATURES
# ============================================================

constant_columns = [
    column
    for column in numeric_data.columns
    if numeric_data[column].nunique() <= 1
]

if constant_columns:
    print("\nConstant features removed:")

    for column in constant_columns:
        print(" -", column)

    numeric_data = numeric_data.drop(
        columns=constant_columns
    )
else:
    print("\nNo constant features found.")

if numeric_data.shape[1] == 0:
    raise ValueError(
        "No usable numeric features remain."
    )


# ============================================================
# 8. STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

X = scaler.fit_transform(numeric_data)

print("\nFeature standardization completed.")
print("Feature matrix shape:", X.shape)


# ============================================================
# 9. CREATE AND TRAIN ISOLATION FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING ISOLATION FOREST")
print("=" * 70)

isolation_forest = IsolationForest(
    n_estimators=200,
    contamination="auto",
    max_samples="auto",
    max_features=1.0,
    bootstrap=False,
    random_state=42,
    n_jobs=-1
)

isolation_forest.fit(X)

print("Isolation Forest training completed.")


# ============================================================
# 10. PREDICT ANOMALIES
# ============================================================

# Isolation Forest output:
#  1 = Normal
# -1 = Anomaly

raw_prediction = isolation_forest.predict(X)

# Easy-to-understand labels:
#  0 = Normal
#  1 = Anomaly

anomaly_label = np.where(
    raw_prediction == -1,
    1,
    0
)


# ============================================================
# 11. CALCULATE ANOMALY SCORES
# ============================================================

# Lower score_samples values indicate more abnormal records.
# Negating the score makes higher values indicate more abnormal.

raw_scores = isolation_forest.score_samples(X)

anomaly_score = -raw_scores


# ============================================================
# 12. ADD RESULTS TO A COPY OF THE ORIGINAL DATA
# ============================================================

results = data.copy()

# Add unique record identifier.
results.insert(
    0,
    "Record_ID",
    np.arange(1, len(results) + 1)
)

results["IsolationForest_Prediction"] = raw_prediction

results["Anomaly_Label"] = anomaly_label

results["Anomaly_Score"] = anomaly_score

results["Anomaly_Status"] = np.where(
    results["Anomaly_Label"] == 1,
    "Anomaly",
    "Normal"
)


# ============================================================
# 13. CALCULATE ANOMALY STATISTICS
# ============================================================

total_records = len(results)

normal_count = int(
    (results["Anomaly_Label"] == 0).sum()
)

anomaly_count = int(
    (results["Anomaly_Label"] == 1).sum()
)

normal_percentage = (
    normal_count / total_records * 100
)

anomaly_percentage = (
    anomaly_count / total_records * 100
)

print("\n" + "=" * 70)
print("SKILL MATCH ANOMALY DETECTION RESULTS")
print("=" * 70)

print("\nTotal Records:", total_records)

print(
    "Normal Records:",
    normal_count,
    f"({normal_percentage:.2f}%)"
)

print(
    "Anomalous Records:",
    anomaly_count,
    f"({anomaly_percentage:.2f}%)"
)


# ============================================================
# 14. SAVE COMPLETE RESULTS
# ============================================================

results_file = os.path.join(
    OUTPUT_FOLDER,
    "isolation_forest_results.csv"
)

results.to_csv(
    results_file,
    index=False
)

print("\nComplete results saved:", results_file)


# ============================================================
# 15. SAVE ONLY ANOMALOUS RECORDS
# ============================================================

anomalies = results[
    results["Anomaly_Label"] == 1
].copy()

anomaly_file = os.path.join(
    OUTPUT_FOLDER,
    "detected_anomalies.csv"
)

anomalies.to_csv(
    anomaly_file,
    index=False
)

print("Anomalous records saved:", anomaly_file)


# ============================================================
# 16. SAVE ONLY NORMAL RECORDS
# ============================================================

normal_records = results[
    results["Anomaly_Label"] == 0
].copy()

normal_file = os.path.join(
    OUTPUT_FOLDER,
    "normal_records.csv"
)

normal_records.to_csv(
    normal_file,
    index=False
)

print("Normal records saved:", normal_file)


# ============================================================
# 17. SAVE ANOMALY SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Metric": [
        "Total Records",
        "Numeric Features Used",
        "Constant Features Removed",
        "Missing Values Before Processing",
        "Missing Values After Processing",
        "Normal Records",
        "Normal Percentage",
        "Anomalous Records",
        "Anomaly Percentage",
        "Isolation Forest Estimators",
        "Random State"
    ],
    "Value": [
        total_records,
        numeric_data.shape[1],
        len(constant_columns),
        missing_before,
        missing_after,
        normal_count,
        f"{normal_percentage:.2f}%",
        anomaly_count,
        f"{anomaly_percentage:.2f}%",
        200,
        42
    ]
})

summary_file = os.path.join(
    OUTPUT_FOLDER,
    "performance_results.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

print("Summary saved:", summary_file)


# ============================================================
# 18. SAVE FEATURE INFORMATION
# ============================================================

feature_information = pd.DataFrame({
    "Feature_Name": numeric_data.columns
})

feature_file = os.path.join(
    OUTPUT_FOLDER,
    "features_used.csv"
)

feature_information.to_csv(
    feature_file,
    index=False
)


# ============================================================
# 19. SAVE ANOMALY SCORE STATISTICS
# ============================================================

score_statistics = pd.DataFrame({
    "Statistic": [
        "Minimum Anomaly Score",
        "Maximum Anomaly Score",
        "Mean Anomaly Score",
        "Median Anomaly Score",
        "Standard Deviation"
    ],
    "Value": [
        np.min(anomaly_score),
        np.max(anomaly_score),
        np.mean(anomaly_score),
        np.median(anomaly_score),
        np.std(anomaly_score)
    ]
})

score_file = os.path.join(
    OUTPUT_FOLDER,
    "anomaly_score_statistics.csv"
)

score_statistics.to_csv(
    score_file,
    index=False
)


# ============================================================
# 20. SAVE TOP 100 ANOMALOUS RECORDS
# ============================================================

top_anomalies = results.sort_values(
    by="Anomaly_Score",
    ascending=False
).head(100)

top_anomaly_file = os.path.join(
    OUTPUT_FOLDER,
    "top_100_anomalous_records.csv"
)

top_anomalies.to_csv(
    top_anomaly_file,
    index=False
)


# ============================================================
# 21. CHART 1 - NORMAL VS ANOMALY
# ============================================================

labels = ["Normal", "Anomaly"]

counts = [normal_count, anomaly_count]

plt.figure(figsize=(8, 6))

bars = plt.bar(labels, counts)

plt.title(
    "Skill Match - Normal vs Anomaly"
)

plt.xlabel("Record Classification")
plt.ylabel("Number of Employee Records")

for bar, count in zip(bars, counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(count),
        ha="center",
        va="bottom"
    )

plt.tight_layout()

chart1 = os.path.join(
    OUTPUT_FOLDER,
    "normal_vs_anomaly.png"
)

plt.savefig(
    chart1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 22. CHART 2 - ANOMALY SCORE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    anomaly_score,
    bins=50
)

plt.title(
    "Skill Match - Anomaly Score Distribution"
)

plt.xlabel("Anomaly Score")
plt.ylabel("Number of Employee Records")

plt.tight_layout()

chart2 = os.path.join(
    OUTPUT_FOLDER,
    "anomaly_score_distribution.png"
)

plt.savefig(
    chart2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 23. CHART 3 - ANOMALY SCORE BY RECORD
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    np.arange(1, total_records + 1),
    anomaly_score,
    linewidth=0.8
)

plt.title(
    "Skill Match - Anomaly Score by Employee Record"
)

plt.xlabel("Record ID")
plt.ylabel("Anomaly Score")

plt.tight_layout()

chart3 = os.path.join(
    OUTPUT_FOLDER,
    "anomaly_score_by_record.png"
)

plt.savefig(
    chart3,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 24. CHART 4 - NORMAL VS ANOMALY PERCENTAGE
# ============================================================

percentages = [
    normal_percentage,
    anomaly_percentage
]

plt.figure(figsize=(8, 6))

bars = plt.bar(
    labels,
    percentages
)

plt.title(
    "Skill Match - Normal and Anomaly Percentage"
)

plt.xlabel("Classification")
plt.ylabel("Percentage (%)")

for bar, value in zip(bars, percentages):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

chart4 = os.path.join(
    OUTPUT_FOLDER,
    "normal_anomaly_percentage.png"
)

plt.savefig(
    chart4,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 25. SAVE TRAINED MODEL AND SCALER
# ============================================================

# Save both objects so new employee data can be transformed
# with the same scaler and evaluated with the same model.

model_file = os.path.join(
    OUTPUT_FOLDER,
    "isolation_forest_model.pkl"
)

with open(model_file, "wb") as file:
    pickle.dump(isolation_forest, file)

scaler_file = os.path.join(
    OUTPUT_FOLDER,
    "standard_scaler.pkl"
)

with open(scaler_file, "wb") as file:
    pickle.dump(scaler, file)


# ============================================================
# 26. SAVE MODEL CONFIGURATION
# ============================================================

model_configuration = pd.DataFrame({
    "Parameter": [
        "Algorithm",
        "Number of Estimators",
        "Contamination",
        "Max Samples",
        "Max Features",
        "Bootstrap",
        "Random State",
        "Feature Scaling",
        "Attrition Target Used as Feature"
    ],
    "Value": [
        "Isolation Forest",
        200,
        "auto",
        "auto",
        1.0,
        False,
        42,
        "StandardScaler",
        False
    ]
})

configuration_file = os.path.join(
    OUTPUT_FOLDER,
    "model_configuration.csv"
)

model_configuration.to_csv(
    configuration_file,
    index=False
)


# ============================================================
# 27. FINAL OUTPUT LIST
# ============================================================

print("\n" + "=" * 70)
print("SKILL MATCH ANOMALY DETECTION COMPLETED")
print("=" * 70)

print("\nOriginal dataset was NOT modified.")

print("\nDataset used:")
print(INPUT_FILE)

print("\nAll outputs stored in:")
print(OUTPUT_FOLDER)

print("\nGenerated files:")

for filename in sorted(os.listdir(OUTPUT_FOLDER)):
    print(" -", filename)

print("\n" + "=" * 70)
print("PROGRAM FINISHED")
print("=" * 70)
