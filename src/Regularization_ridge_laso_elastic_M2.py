
# ============================================================
# SKILL_MATCH - REGULARISATION
# Ridge (L2), Lasso (L1), Elastic Net
#
# IMPORTANT:
# 1. The original dataset is NOT modified.
# 2. All processing is performed on copies.
# 3. Categorical features are encoded into numeric values.
# 4. Feature scaling is fitted using training data only.
# 5. Graphs are saved as PNG files; plt.show() is not used.
# ============================================================

import os
import time
import pandas as pd
import numpy as np
import matplotlib

# Non-interactive backend: saves plots without display warnings.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# 1. FILE SETTINGS
# ============================================================

start_time = time.time()

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

FILE_NAME = os.path.join(
    BASE_DIR,
    "dataset",
    "reordered_raw.csv"
)

output_folder = os.path.join(
    BASE_DIR,
    "output",
    "Regularization_Ridge_Lasso_ElasticNet_Outputs"
)

os.makedirs(output_folder, exist_ok=True)

# Current IBM HR dataset uses Attrition.
TARGET_COLUMN = "Attrition"


# ============================================================
# 2. READ PREPROCESSED DATASET
# ============================================================

if not os.path.exists(FILE_NAME):
    raise FileNotFoundError(
        f"Dataset not found: {FILE_NAME}"
    )

df_original = pd.read_csv(FILE_NAME)

print("\n================================================")
print("ORIGINAL DATASET")
print("================================================")

print(df_original.head())
print("\nShape:", df_original.shape)
print("\nColumns:")
print(df_original.columns.tolist())


# ============================================================
# 3. CREATE A COPY
# ============================================================

df = df_original.copy(deep=True)


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

print("\n================================================")
print("MISSING VALUE CHECK")
print("================================================")

print(df.isnull().sum())

df_model = df.copy(deep=True)


# ============================================================
# 5. CHECK TARGET COLUMN
# ============================================================

if TARGET_COLUMN not in df_model.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' does not exist.\n"
        f"Available columns:\n{df_model.columns.tolist()}"
    )


# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

X = df_model.drop(
    columns=[TARGET_COLUMN]
).copy()

y = df_model[TARGET_COLUMN].copy()


# ============================================================
# 7. CONVERT TARGET TO 0 AND 1
# ============================================================

print("\n================================================")
print("TARGET VALUES")
print("================================================")

print(y.unique())

if (
    pd.api.types.is_object_dtype(y)
    or isinstance(y.dtype, pd.StringDtype)
    or pd.api.types.is_categorical_dtype(y)
):
    y = y.astype(str).str.strip().str.lower()

    target_mapping = {
        "yes": 1,
        "no": 0,
        "placed": 1,
        "not placed": 0,
        "true": 1,
        "false": 0
    }

    y = y.map(target_mapping)

elif pd.api.types.is_bool_dtype(y):
    y = y.astype(int)

else:
    unique_values = sorted(y.dropna().unique())

    if set(unique_values) != {0, 1}:
        if len(unique_values) == 2:
            mapping = {
                unique_values[0]: 0,
                unique_values[1]: 1
            }
            y = y.map(mapping)

if y.isnull().any():
    raise ValueError(
        "Target values could not be converted to 0 and 1. "
        "Check the target column's values."
    )

y = y.astype(int)

if y.nunique() != 2:
    raise ValueError(
        "The target must contain exactly two classes."
    )

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 8. ENCODE CATEGORICAL FEATURES
# ============================================================

print("\n================================================")
print("FEATURE PREPROCESSING")
print("================================================")

# Convert categorical columns to numeric dummy variables.
X = pd.get_dummies(
    X,
    drop_first=True,
    dtype=int
)

# Replace infinite values with missing values.
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

# Remove columns that contain only missing values.
X = X.dropna(
    axis=1,
    how="all"
)

# Fill remaining missing values using column medians.
X = X.fillna(
    X.median()
)

if X.shape[1] == 0:
    raise ValueError(
        "No usable predictor columns remain."
    )

if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError(
        "Features contain invalid numeric values after preprocessing."
    )

print("All predictor variables are numeric.")
print("Number of features:", X.shape[1])


# ============================================================
# 9. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n================================================")
print("TRAIN-TEST SPLIT")
print("================================================")

print("Training records:", len(X_train))
print("Testing records :", len(X_test))


# ============================================================
# 10. RIDGE REGULARISATION - L2
# ============================================================

print("\n================================================")
print("RIDGE REGULARISATION - L2")
print("================================================")

ridge = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)

ridge.fit(
    X_train,
    y_train
)

ridge_values = ridge.predict(X_test)

ridge_predictions = (
    ridge_values >= 0.5
).astype(int)

print("Ridge model trained successfully.")


# ============================================================
# 11. LASSO REGULARISATION - L1
# ============================================================

print("\n================================================")
print("LASSO REGULARISATION - L1")
print("================================================")

lasso = make_pipeline(
    StandardScaler(),
    Lasso(
        alpha=0.01,
        max_iter=10000
    )
)

lasso.fit(
    X_train,
    y_train
)

lasso_values = lasso.predict(X_test)

lasso_predictions = (
    lasso_values >= 0.5
).astype(int)

print("Lasso model trained successfully.")


# ============================================================
# 12. ELASTIC NET
# ============================================================

print("\n================================================")
print("ELASTIC NET")
print("================================================")

elastic = make_pipeline(
    StandardScaler(),
    ElasticNet(
        alpha=0.01,
        l1_ratio=0.5,
        max_iter=10000
    )
)

elastic.fit(
    X_train,
    y_train
)

elastic_values = elastic.predict(X_test)

elastic_predictions = (
    elastic_values >= 0.5
).astype(int)

print("Elastic Net model trained successfully.")


# ============================================================
# 13. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    y_actual,
    y_predicted
):
    accuracy = accuracy_score(
        y_actual,
        y_predicted
    )

    precision = precision_score(
        y_actual,
        y_predicted,
        zero_division=0
    )

    recall = recall_score(
        y_actual,
        y_predicted,
        zero_division=0
    )

    f1 = f1_score(
        y_actual,
        y_predicted,
        zero_division=0
    )

    print("\n--------------------------------------------")
    print(model_name)
    print("--------------------------------------------")

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("\nClassification Report:")

    print(
        classification_report(
            y_actual,
            y_predicted,
            zero_division=0
        )
    )

    return [
        accuracy,
        precision,
        recall,
        f1
    ]


# ============================================================
# 14. EVALUATE RIDGE
# ============================================================

ridge_results = evaluate_model(
    "Ridge (L2)",
    y_test,
    ridge_predictions
)


# ============================================================
# 15. EVALUATE LASSO
# ============================================================

lasso_results = evaluate_model(
    "Lasso (L1)",
    y_test,
    lasso_predictions
)


# ============================================================
# 16. EVALUATE ELASTIC NET
# ============================================================

elastic_results = evaluate_model(
    "Elastic Net",
    y_test,
    elastic_predictions
)


# ============================================================
# 17. CREATE MODEL COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "Ridge (L2)",
        "Lasso (L1)",
        "Elastic Net"
    ],
    "Accuracy": [
        ridge_results[0],
        lasso_results[0],
        elastic_results[0]
    ],
    "Precision": [
        ridge_results[1],
        lasso_results[1],
        elastic_results[1]
    ],
    "Recall": [
        ridge_results[2],
        lasso_results[2],
        elastic_results[2]
    ],
    "F1_Score": [
        ridge_results[3],
        lasso_results[3],
        elastic_results[3]
    ]
})

print("\n================================================")
print("REGULARISATION MODEL COMPARISON")
print("================================================")

print(comparison)


# ============================================================
# 18. SAVE MODEL COMPARISON
# ============================================================

comparison_path = os.path.join(
    output_folder,
    "regularisation_model_comparison_M2.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)

print("\nSaved:", comparison_path)


# ============================================================
# 19. CREATE COEFFICIENT TABLE
# ============================================================

# Extract the fitted models from their pipelines.
ridge_model = ridge.named_steps["ridge"]
lasso_model = lasso.named_steps["lasso"]
elastic_model = elastic.named_steps["elasticnet"]

coefficient_table = pd.DataFrame({
    "Feature": X.columns,
    "Ridge_L2": ridge_model.coef_,
    "Lasso_L1": lasso_model.coef_,
    "Elastic_Net": elastic_model.coef_
})

print("\n================================================")
print("REGULARISATION COEFFICIENTS")
print("================================================")

print(coefficient_table)


# ============================================================
# 20. SAVE COEFFICIENT TABLE
# ============================================================

coefficient_path = os.path.join(
    output_folder,
    "regularisation_coefficients_M2.csv"
)

coefficient_table.to_csv(
    coefficient_path,
    index=False
)

print("\nSaved:", coefficient_path)


# ============================================================
# 21. IDENTIFY LASSO ZERO COEFFICIENTS
# ============================================================

zero_lasso = coefficient_table[
    np.isclose(
        coefficient_table["Lasso_L1"],
        0,
        atol=1e-6
    )
].copy()

print("\n================================================")
print("LASSO SPARSITY / FEATURE SELECTION")
print("================================================")

print("Total features:", len(X.columns))
print("Zero Lasso coefficients:", len(zero_lasso))
print(
    "Non-zero Lasso coefficients:",
    len(X.columns) - len(zero_lasso)
)

if len(zero_lasso) > 0:
    print(
        "\nFeatures whose Lasso coefficient became zero:"
    )

    print(
        zero_lasso[
            ["Feature", "Lasso_L1"]
        ].to_string(index=False)
    )
else:
    print("\nNo coefficients became exactly zero.")
    print(
        "Increase alpha if stronger L1 regularisation is required."
    )


# ============================================================
# 22. SAVE LASSO FEATURE SELECTION RESULT
# ============================================================

lasso_features = coefficient_table.copy()

lasso_features["Selected_by_Lasso"] = (
    np.abs(lasso_features["Lasso_L1"]) > 1e-6
)

lasso_features_path = os.path.join(
    output_folder,
    "lasso_feature_selection_M2.csv"
)

lasso_features.to_csv(
    lasso_features_path,
    index=False
)

print("\nSaved:", lasso_features_path)


# ============================================================
# 23. GEOMETRIC PICTURE - L1 VS L2
# ============================================================

beta1 = np.linspace(-2, 2, 500)
beta2 = np.linspace(-2, 2, 500)

BETA1, BETA2 = np.meshgrid(
    beta1,
    beta2
)

# L1 constraint: |beta1| + |beta2| = C
C = 1.5

L1 = (
    np.abs(BETA1)
    + np.abs(BETA2)
)

# L2 constraint: beta1^2 + beta2^2 = C^2
L2 = (
    BETA1 ** 2
    + BETA2 ** 2
)

plt.figure(figsize=(8, 8))

plt.contour(
    BETA1,
    BETA2,
    L1,
    levels=[C],
    linewidths=2
)

plt.contour(
    BETA1,
    BETA2,
    L2,
    levels=[C ** 2],
    linewidths=2
)

plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)

plt.xlabel("Coefficient beta1")
plt.ylabel("Coefficient beta2")
plt.title("Geometric Picture of L1 and L2 Regularisation")

plt.text(1.15, 0.05, "L1\nDiamond", fontsize=12)
plt.text(0.45, 1.25, "L2\nCircle", fontsize=12)

plt.grid(True)
plt.axis("equal")
plt.tight_layout()

image_path = os.path.join(
    output_folder,
    "L1_vs_L2_Regularisation.png"
)

plt.savefig(
    image_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:", image_path)


# ============================================================
# 24. LASSO SPARSITY GRAPH
# ============================================================

plt.figure(figsize=(12, 6))

feature_numbers = np.arange(len(X.columns))

plt.stem(
    feature_numbers,
    lasso_model.coef_
)

plt.axhline(0, linewidth=1)

plt.xlabel("Feature Number")
plt.ylabel("Lasso Coefficient")
plt.title("Lasso L1 Regularisation - Feature Sparsity")

plt.xticks(
    feature_numbers,
    X.columns,
    rotation=90
)

plt.grid(True)
plt.tight_layout()

image_path = os.path.join(
    output_folder,
    "Lasso_Sparsity_Graph.png"
)

plt.savefig(
    image_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:", image_path)


# ============================================================
# 25. COEFFICIENT COMPARISON GRAPH
# ============================================================

x = np.arange(len(X.columns))
width = 0.25

plt.figure(figsize=(14, 7))

plt.bar(
    x - width,
    ridge_model.coef_,
    width,
    label="Ridge (L2)"
)

plt.bar(
    x,
    lasso_model.coef_,
    width,
    label="Lasso (L1)"
)

plt.bar(
    x + width,
    elastic_model.coef_,
    width,
    label="Elastic Net"
)

plt.axhline(0, linewidth=1)

plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Ridge vs Lasso vs Elastic Net Coefficients")

plt.xticks(
    x,
    X.columns,
    rotation=90
)

plt.legend()
plt.tight_layout()

image_path = os.path.join(
    output_folder,
    "Coefficient_Comparison.png"
)

plt.savefig(
    image_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:", image_path)


# ============================================================
# 26. MODEL PERFORMANCE GRAPH
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

x = np.arange(len(metrics))
width = 0.25

plt.figure(figsize=(10, 6))

plt.bar(
    x - width,
    ridge_results,
    width,
    label="Ridge"
)

plt.bar(
    x,
    lasso_results,
    width,
    label="Lasso"
)

plt.bar(
    x + width,
    elastic_results,
    width,
    label="Elastic Net"
)

plt.xticks(x, metrics)
plt.ylim(0, 1.1)

plt.ylabel("Score")
plt.title("Regularisation Model Performance")

plt.legend()
plt.grid(axis="y")
plt.tight_layout()

image_path = os.path.join(
    output_folder,
    "Model_Performance.png"
)

plt.savefig(
    image_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:", image_path)


# ============================================================
# 27. VERIFY ORIGINAL DATASET WAS NOT CHANGED
# ============================================================

execution_time = time.time() - start_time

print("\n================================================")
print("VERIFY ORIGINAL DATASET")
print("================================================")

print("Original dataset shape:", df_original.shape)
print("Original records:", len(df_original))

print("\nThe original dataset was NOT modified.")

print("\n================================================")
print("PROGRAM COMPLETED SUCCESSFULLY")
print("================================================")

print("Execution time:", round(execution_time, 2), "seconds")
print("Output folder:", output_folder)

print("\nGenerated output files:")

for filename in sorted(os.listdir(output_folder)):
    print(filename)

print("\nAll results and graphs have been saved successfully!")
