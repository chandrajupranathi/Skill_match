
# ============================================================
# SKILL MATCH
# LINEAR REGRESSION
# Closed-Form Normal Equation vs Gradient Descent
#
# Dataset: reordered_raw.csv
# Target: Attrition (No = 0, Yes = 1)
# Images are stored in ONE separate folder
# ============================================================

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# 1. FILE PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\chand\Downloads\Skill_Match (2)\Skill_Match"
)

DATASET_PATH = BASE_DIR / "dataset" / "reordered_raw.csv"

OUTPUT_FOLDER = (
    BASE_DIR / "output" / "Linear_Regression_CFNE_GD_Compare_M2"
)

IMAGE_FOLDER = OUTPUT_FOLDER / "images"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("SKILL MATCH")
print("LINEAR REGRESSION")
print("NORMAL EQUATION VS GRADIENT DESCENT")
print("=" * 65)


# ============================================================
# 2. CHECK AND LOAD DATASET
# ============================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

data = pd.read_csv(DATASET_PATH, low_memory=False)

print("\nOriginal Dataset Shape:", data.shape)
print("\nFirst 5 Records:")
print(data.head())

data.columns = data.columns.astype(str).str.strip()

print("\nDataset Columns:")
print(list(data.columns))


# ============================================================
# 3. SELECT TARGET
# ============================================================

TARGET_COLUMN = "Attrition"

if TARGET_COLUMN not in data.columns:
    raise ValueError(
        f"Target '{TARGET_COLUMN}' not found.\n"
        f"Available columns: {list(data.columns)}"
    )

# Convert target values into numeric values:
# No = 0, Yes = 1

data[TARGET_COLUMN] = (
    data[TARGET_COLUMN]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"no": 0, "yes": 1})
)

data = data.dropna(subset=[TARGET_COLUMN]).copy()
data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)

print("\nTarget Column:", TARGET_COLUMN)
print("\nTarget Distribution:")
print(data[TARGET_COLUMN].value_counts())


# ============================================================
# 4. SELECT FEATURES
# ============================================================

# Exclude target, identifiers, and constant columns.

excluded_columns = [
    TARGET_COLUMN,
    "EmployeeNumber",
    "EmployeeCount",
    "StandardHours",
    "Over18"
]

X_raw = data.drop(
    columns=excluded_columns,
    errors="ignore"
).copy()

y = data[TARGET_COLUMN].copy()

# Remove columns that contain only missing values.

X_raw = X_raw.dropna(axis=1, how="all")

# Remove constant columns.

constant_columns = [
    column for column in X_raw.columns
    if X_raw[column].nunique(dropna=True) <= 1
]

if constant_columns:
    print("\nRemoving constant columns:", constant_columns)
    X_raw = X_raw.drop(columns=constant_columns)

print("\nOriginal Feature Columns:")
print(list(X_raw.columns))


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 65)
print("TRAIN-TEST SPLIT")
print("=" * 65)

print("Training samples:", len(X_train_raw))
print("Testing samples :", len(X_test_raw))


# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

# Learn imputation values from training data only.

for column in X_train_raw.columns:

    if (
        pd.api.types.is_numeric_dtype(X_train_raw[column])
        and not pd.api.types.is_bool_dtype(X_train_raw[column])
    ):
        fill_value = X_train_raw[column].median()

        if pd.isna(fill_value):
            fill_value = 0

    else:
        modes = X_train_raw[column].mode(dropna=True)

        fill_value = modes.iloc[0] if not modes.empty else "Unknown"

    X_train_raw[column] = X_train_raw[column].fillna(fill_value)
    X_test_raw[column] = X_test_raw[column].fillna(fill_value)


# ============================================================
# 7. ENCODE CATEGORICAL FEATURES
# ============================================================

# Convert text columns into numeric one-hot encoded columns.

X_train = pd.get_dummies(
    X_train_raw,
    drop_first=True,
    dtype=int
)

X_test = pd.get_dummies(
    X_test_raw,
    drop_first=True,
    dtype=int
)

# Ensure training and testing have identical feature columns.

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

feature_columns = X_train.columns.tolist()

# Convert features to floating-point numbers.

X_train = X_train.astype(float)
X_test = X_test.astype(float)

# Remove any remaining constant features based on training data.

varying_columns = X_train.columns[
    X_train.nunique(dropna=False) > 1
]

X_train = X_train[varying_columns]
X_test = X_test[varying_columns]

feature_columns = X_train.columns.tolist()

if X_train.shape[1] == 0:
    raise ValueError("No usable feature columns remain.")

print("\nNumber of Encoded Features:", len(feature_columns))
print("\nEncoded Feature Names:")
print(feature_columns)


# ============================================================
# 8. SAVE CLEANED MODEL DATA
# ============================================================

cleaned_data = pd.concat(
    [
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True).rename(TARGET_COLUMN)
    ],
    axis=1
)

cleaned_data.to_csv(
    OUTPUT_FOLDER / "skill_match_cleaned_data.csv",
    index=False
)

print("\nCleaned training data saved.")


# ============================================================
# 9. FEATURE SCALING
# ============================================================

# Use the same scaled features for BOTH methods.
# This makes their comparison fair.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_array = y_train.to_numpy(dtype=float)
y_test_array = y_test.to_numpy(dtype=float)

print("\nTraining feature shape:", X_train_scaled.shape)
print("Testing feature shape :", X_test_scaled.shape)


# ============================================================
# 10. ADD BIAS / INTERCEPT COLUMN
# ============================================================

X_train_bias = np.c_[
    np.ones(X_train_scaled.shape[0]),
    X_train_scaled
]

X_test_bias = np.c_[
    np.ones(X_test_scaled.shape[0]),
    X_test_scaled
]


# ============================================================
# 11. CLOSED-FORM NORMAL EQUATION
# ============================================================

print("\n" + "=" * 65)
print("CLOSED-FORM NORMAL EQUATION")
print("=" * 65)

# theta = pseudoinverse(X) * y

theta_normal = (
    np.linalg.pinv(X_train_bias) @ y_train_array
)

pred_normal = X_test_bias @ theta_normal

mse_normal = mean_squared_error(
    y_test_array,
    pred_normal
)

rmse_normal = np.sqrt(mse_normal)

r2_normal = r2_score(
    y_test_array,
    pred_normal
)

print("\nNormal Equation Intercept:", theta_normal[0])
print("Normal Equation MSE:", mse_normal)
print("Normal Equation RMSE:", rmse_normal)
print("Normal Equation R2:", r2_normal)


# ============================================================
# 12. GRADIENT DESCENT
# ============================================================

print("\n" + "=" * 65)
print("GRADIENT DESCENT")
print("=" * 65)

m = len(y_train_array)

theta_gd = np.zeros(X_train_bias.shape[1])

learning_rate = 0.01
epochs = 1000

print("\nLearning Rate:", learning_rate)
print("Epochs:", epochs)

loss_history = []

# Gradient Descent iterations.

for epoch in range(epochs):

    predictions = X_train_bias @ theta_gd

    errors = predictions - y_train_array

    gradients = (2 / m) * (X_train_bias.T @ errors)

    theta_gd -= learning_rate * gradients

    updated_predictions = X_train_bias @ theta_gd

    loss = np.mean(
        (updated_predictions - y_train_array) ** 2
    )

    loss_history.append(loss)


# ============================================================
# 13. GRADIENT DESCENT PREDICTIONS AND METRICS
# ============================================================

pred_gd = X_test_bias @ theta_gd

mse_gd = mean_squared_error(
    y_test_array,
    pred_gd
)

rmse_gd = np.sqrt(mse_gd)

r2_gd = r2_score(
    y_test_array,
    pred_gd
)

print("\nGradient Descent Intercept:", theta_gd[0])
print("Gradient Descent MSE:", mse_gd)
print("Gradient Descent RMSE:", rmse_gd)
print("Gradient Descent R2:", r2_gd)


# ============================================================
# 14. COMPARE BOTH METHODS
# ============================================================

print("\n" + "=" * 65)
print("NORMAL EQUATION VS GRADIENT DESCENT")
print("=" * 65)

comparison_df = pd.DataFrame({
    "Method": [
        "Normal Equation",
        "Gradient Descent"
    ],
    "MSE": [
        mse_normal,
        mse_gd
    ],
    "RMSE": [
        rmse_normal,
        rmse_gd
    ],
    "R2": [
        r2_normal,
        r2_gd
    ]
})

print("\n", comparison_df.to_string(index=False))

comparison_df.to_csv(
    OUTPUT_FOLDER / "normal_equation_vs_gradient_descent.csv",
    index=False
)


# ============================================================
# 15. IMAGE 1 - ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test_array,
    pred_normal,
    alpha=0.6,
    label="Normal Equation"
)

plt.scatter(
    y_test_array,
    pred_gd,
    alpha=0.6,
    label="Gradient Descent"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Attrition (0 = No, 1 = Yes)")
plt.ylabel("Predicted Attrition Value")
plt.title("Attrition - Actual vs Predicted Values")
plt.legend()
plt.grid(True)
plt.tight_layout()

image1 = IMAGE_FOLDER / "actual_vs_predicted.png"

plt.savefig(image1, dpi=300, bbox_inches="tight")
plt.close()

print("\nImage 1 saved:", image1)


# ============================================================
# 16. IMAGE 2 - RESIDUAL COMPARISON
# ============================================================

normal_residuals = y_test_array - pred_normal
gd_residuals = y_test_array - pred_gd

plt.figure(figsize=(9, 6))

plt.scatter(
    pred_normal,
    normal_residuals,
    alpha=0.6,
    label="Normal Equation"
)

plt.scatter(
    pred_gd,
    gd_residuals,
    alpha=0.6,
    label="Gradient Descent"
)

plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Attrition Value")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Attrition - Residual Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()

image2 = IMAGE_FOLDER / "residual_comparison.png"

plt.savefig(image2, dpi=300, bbox_inches="tight")
plt.close()

print("Image 2 saved:", image2)


# ============================================================
# 17. IMAGE 3 - GRADIENT DESCENT LOSS
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    range(1, epochs + 1),
    loss_history
)

plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Gradient Descent Convergence")
plt.grid(True)
plt.tight_layout()

image3 = IMAGE_FOLDER / "gradient_descent_loss.png"

plt.savefig(image3, dpi=300, bbox_inches="tight")
plt.close()

print("Image 3 saved:", image3)


# ============================================================
# 18. SAVE IMAGE INFORMATION
# ============================================================

image_info = pd.DataFrame({
    "Image": [
        "actual_vs_predicted.png",
        "residual_comparison.png",
        "gradient_descent_loss.png"
    ],
    "Description": [
        "Actual versus predicted attrition values",
        "Residual comparison between both methods",
        "Training MSE across Gradient Descent epochs"
    ]
})

image_info.to_csv(
    IMAGE_FOLDER / "image_information.csv",
    index=False
)


# ============================================================
# 19. SAVE COEFFICIENTS
# ============================================================

coefficient_df = pd.DataFrame({
    "Feature": ["Intercept"] + feature_columns,
    "Normal_Equation": theta_normal,
    "Gradient_Descent": theta_gd
})

coefficient_df.to_csv(
    OUTPUT_FOLDER / "linear_regression_coefficients.csv",
    index=False
)


# ============================================================
# 20. SAVE FINAL METRICS
# ============================================================

metrics_df = comparison_df.copy()

metrics_df.to_csv(
    OUTPUT_FOLDER / "linear_regression_metrics.csv",
    index=False
)


# ============================================================
# 21. SAVE TEST PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({
    "Actual_Attrition": y_test_array.astype(int),
    "Normal_Equation_Prediction": pred_normal,
    "Gradient_Descent_Prediction": pred_gd,
    "Normal_Equation_Residual": normal_residuals,
    "Gradient_Descent_Residual": gd_residuals
})

predictions_df.to_csv(
    OUTPUT_FOLDER / "linear_regression_test_predictions.csv",
    index=False
)


# ============================================================
# 22. SAVE FEATURE LIST
# ============================================================

pd.DataFrame({
    "Feature": feature_columns
}).to_csv(
    OUTPUT_FOLDER / "feature_columns.csv",
    index=False
)


# ============================================================
# 23. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 65)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nDataset used:")
print(DATASET_PATH)

print("\nOutput folder:")
print(OUTPUT_FOLDER)

print("\nImages folder:")
print(IMAGE_FOLDER)

print("\nGenerated files:")
print("1. skill_match_cleaned_data.csv")
print("2. normal_equation_vs_gradient_descent.csv")
print("3. linear_regression_coefficients.csv")
print("4. linear_regression_metrics.csv")
print("5. linear_regression_test_predictions.csv")
print("6. feature_columns.csv")
print("7. images/actual_vs_predicted.png")
print("8. images/residual_comparison.png")
print("9. images/gradient_descent_loss.png")
print("10. images/image_information.csv")

print("\nOriginal raw dataset was NOT modified.")
print("Target: Attrition (No = 0, Yes = 1)")
print("\nNOTE: Linear Regression is used for comparison only.")
print("Logistic Regression is more suitable for Yes/No prediction.")

print("\n" + "=" * 65)
