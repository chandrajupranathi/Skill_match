
# ================================================================
# LINEAR REGRESSION - SKILL MATCH
# REGRESSION USING EXISTING DATASET
# ================================================================
# NOTE:
# The current dataset is IBM HR Attrition data.
# Linear Regression predicts a numeric target, so Attrition is
# encoded as No = 0 and Yes = 1.
# This is a regression demonstration, not a true skill-match model.
# The original dataset is never modified.
# ================================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# ================================================================
# 1. PATH CONFIGURATION
# ================================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_FILE = os.path.join(
    BASE_DIR, "dataset", "reordered_raw.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "output", "Linear_Regression_with_Metrics_M2"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COLUMN = "Attrition"
RANDOM_STATE = 42
TEST_SIZE = 0.20

print("=" * 75)
print("LINEAR REGRESSION - SKILL MATCH")
print("=" * 75)

# ================================================================
# 2. LOAD DATASET
# ================================================================

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_FILE}"
    )

data = pd.read_csv(DATASET_FILE)

print("\nDataset loaded successfully.")
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])

# ================================================================
# 3. DISPLAY DATASET INFORMATION
# ================================================================

print("\nFirst 5 rows:")
print(data.head())

print("\nColumn names:")
print(data.columns.tolist())

print("\nData types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isnull().sum())

# ================================================================
# 4. VALIDATE TARGET
# ================================================================

if TARGET_COLUMN not in data.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found."
    )

data = data.dropna(subset=[TARGET_COLUMN]).copy()

# Encode target: No = 0, Yes = 1
data[TARGET_COLUMN] = (
    data[TARGET_COLUMN]
    .astype(str)
    .str.strip()
    .map({"No": 0, "Yes": 1})
)

data = data.dropna(subset=[TARGET_COLUMN]).copy()
data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)

print("\nTarget column:", TARGET_COLUMN)
print("\nTarget distribution:")
print(data[TARGET_COLUMN].value_counts().sort_index())

# ================================================================
# 5. SELECT FEATURES
# ================================================================

excluded_columns = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours"
]

excluded_columns = [
    col for col in excluded_columns if col in data.columns
]

X = data.drop(
    columns=[TARGET_COLUMN] + excluded_columns,
    errors="ignore"
)

y = data[TARGET_COLUMN]

print("\nExcluded columns:", excluded_columns)
print("\nNumber of features:", X.shape[1])

# ================================================================
# 6. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ================================================================

numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNumerical columns:")
print(numeric_columns)

print("\nCategorical columns:")
print(categorical_columns)

# ================================================================
# 7. TRAIN-TEST SPLIT
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ================================================================
# 8. NUMERICAL PREPROCESSING
# ================================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# ================================================================
# 9. CATEGORICAL PREPROCESSING
# ================================================================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# ================================================================
# 10. COMBINE PREPROCESSING
# ================================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ]
)

# ================================================================
# 11. CREATE MODEL PIPELINE
# ================================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

# ================================================================
# 12. TRAIN MODEL
# ================================================================

print("\nTraining Linear Regression model...")

model.fit(X_train, y_train)

print("Model training completed successfully.")

# ================================================================
# 13. MAKE PREDICTIONS
# ================================================================

y_pred = model.predict(X_test)

print("\nPredictions completed.")

# ================================================================
# 14. CALCULATE METRICS
# ================================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 75)
print("LINEAR REGRESSION MODEL RESULTS")
print("=" * 75)

print(f"\nMAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")
print(f"R2 (%) : {r2 * 100:.2f}%")

# ================================================================
# 15. SAVE METRICS
# ================================================================

metrics = pd.DataFrame({
    "Metric": [
        "Mean Absolute Error",
        "Mean Squared Error",
        "Root Mean Squared Error",
        "R2 Score",
        "R2 Score (%)"
    ],
    "Value": [
        mae,
        mse,
        rmse,
        r2,
        r2 * 100
    ]
})

metrics.to_csv(
    os.path.join(OUTPUT_DIR, "Linear_Regression_Metrics.csv"),
    index=False
)

# ================================================================
# 16. SAVE PREDICTIONS
# ================================================================

predictions = X_test.reset_index(drop=True).copy()

predictions["Actual_Attrition"] = y_test.reset_index(
    drop=True
)

predictions["Predicted_Attrition_Score"] = y_pred

predictions["Predicted_Class"] = np.where(
    y_pred >= 0.5, "Yes", "No"
)

predictions.to_csv(
    os.path.join(OUTPUT_DIR, "Linear_Regression_Predictions.csv"),
    index=False
)

# ================================================================
# 17. SAVE MODEL COEFFICIENTS
# ================================================================

fitted_preprocessor = model.named_steps["preprocessor"]
fitted_regressor = model.named_steps["regressor"]

feature_names = fitted_preprocessor.get_feature_names_out()

coefficients = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": fitted_regressor.coef_
})

coefficients["Absolute_Coefficient"] = (
    coefficients["Coefficient"].abs()
)

coefficients.sort_values(
    "Absolute_Coefficient",
    ascending=False,
    inplace=True
)

coefficients.to_csv(
    os.path.join(OUTPUT_DIR, "Linear_Regression_Coefficients.csv"),
    index=False
)

# ================================================================
# 18. SAVE INTERCEPT
# ================================================================

intercept = pd.DataFrame({
    "Intercept": [fitted_regressor.intercept_]
})

intercept.to_csv(
    os.path.join(OUTPUT_DIR, "Linear_Regression_Intercept.csv"),
    index=False
)

# ================================================================
# 19. ACTUAL VS PREDICTED GRAPH
# ================================================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.6)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("Actual Attrition (0 = No, 1 = Yes)")
plt.ylabel("Predicted Score")
plt.title("Linear Regression: Actual vs Predicted")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "Actual_vs_Predicted.png"),
    dpi=300
)

plt.close()

# ================================================================
# 20. RESIDUAL GRAPH
# ================================================================

residuals = y_test.to_numpy() - y_pred

plt.figure(figsize=(8, 6))

plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Score")
plt.ylabel("Residual")
plt.title("Linear Regression: Residual Plot")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "Residual_Plot.png"),
    dpi=300
)

plt.close()

# ================================================================
# 21. ERROR DISTRIBUTION
# ================================================================

plt.figure(figsize=(8, 6))

plt.hist(residuals, bins=20, edgecolor="black")

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "Error_Distribution.png"),
    dpi=300
)

plt.close()

# ================================================================
# 22. SAVE TRAINED MODEL
# ================================================================

joblib.dump(
    model,
    os.path.join(OUTPUT_DIR, "Linear_Regression_Model.pkl")
)

# ================================================================
# 23. SAVE CONFIGURATION
# ================================================================

configuration = pd.DataFrame({
    "Parameter": [
        "Target column",
        "Test size",
        "Random state",
        "Algorithm",
        "Training samples",
        "Testing samples",
        "Original feature count",
        "Encoded feature count"
    ],
    "Value": [
        TARGET_COLUMN,
        TEST_SIZE,
        RANDOM_STATE,
        "Linear Regression",
        len(X_train),
        len(X_test),
        X.shape[1],
        len(feature_names)
    ]
})

configuration.to_csv(
    os.path.join(OUTPUT_DIR, "Model_Configuration.csv"),
    index=False
)

# ================================================================
# 24. FINAL SUMMARY
# ================================================================

print("\n" + "=" * 75)
print("ALL OUTPUTS SAVED SUCCESSFULLY")
print("=" * 75)

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("-", filename)

print("\nProgram completed successfully.")

print(
    "\nReminder: this dataset predicts employee attrition. "
    "Linear Regression is being used only as a regression demonstration; "
    "Logistic Regression is more appropriate for Yes/No classification."
)
