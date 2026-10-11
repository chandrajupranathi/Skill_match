
# ================================================================
# LOGISTIC REGRESSION - SKILL MATCH
# CLASSIFICATION USING EXISTING DATASET
# ================================================================
# Dataset: reordered_raw.csv
# Target: Attrition (No / Yes)
# Algorithm: Logistic Regression
# Output: Skill_Match/output/
#
# NOTE:
# The existing dataset predicts employee attrition.
# It does not contain resume skills or placement outcomes.
# ================================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

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
    BASE_DIR,
    "output",
    "Multinomial_Logistic_Regre_for_Multiclass_M2"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COLUMN = "Attrition"
TEST_SIZE = 0.20
RANDOM_STATE = 42

print("=" * 75)
print("LOGISTIC REGRESSION - SKILL MATCH")
print("CLASSIFICATION USING EXISTING DATASET")
print("=" * 75)

# ================================================================
# 2. LOAD DATASET
# ================================================================

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_FILE}"
    )

data = pd.read_csv(DATASET_FILE)

print("\nDataset loaded successfully.")
print("Rows    :", data.shape[0])
print("Columns :", data.shape[1])

# ================================================================
# 3. DISPLAY DATASET INFORMATION
# ================================================================

print("\n" + "=" * 75)
print("EXISTING DATASET")
print("=" * 75)

print("\nFirst 5 rows:")
print(data.head())

print("\nColumn names:")
print(data.columns.tolist())

print("\nData types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isnull().sum())

# ================================================================
# 4. VALIDATE TARGET COLUMN
# ================================================================

if TARGET_COLUMN not in data.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' was not found."
    )

data = data.dropna(subset=[TARGET_COLUMN]).copy()

print("\nTarget column:", TARGET_COLUMN)

# ================================================================
# 5. TARGET CLASS DISTRIBUTION
# ================================================================

print("\n" + "=" * 75)
print("TARGET CLASS DISTRIBUTION")
print("=" * 75)

print(data[TARGET_COLUMN].value_counts())

number_of_classes = data[TARGET_COLUMN].nunique()

print("\nNumber of classes:", number_of_classes)

if number_of_classes < 2:
    raise ValueError("At least two target classes are required.")

if number_of_classes == 2:
    print(
        "\nNote: Attrition has two classes, "
        "so this is binary classification."
    )

# ================================================================
# 6. REMOVE ID AND CONSTANT COLUMNS
# ================================================================

columns_to_exclude = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours"
]

excluded_columns = [
    column for column in columns_to_exclude
    if column in data.columns
]

print("\nExcluded columns:")
print(excluded_columns)

X = data.drop(
    columns=[TARGET_COLUMN] + excluded_columns,
    errors="ignore"
)

y_text = data[TARGET_COLUMN].astype(str).str.strip()

# ================================================================
# 7. FEATURE INFORMATION
# ================================================================

print("\nFeature columns:")

for column in X.columns:
    print("-", column)

print("\nNumber of original features:", X.shape[1])

# ================================================================
# 8. IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
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

print("\nMissing values in features:")
print(X.isnull().sum())

print(
    "\nTotal missing feature values:",
    int(X.isnull().sum().sum())
)

# ================================================================
# 9. ENCODE TARGET
# ================================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y_text)

class_names = label_encoder.classes_.tolist()

print("\nTarget encoding completed.")

print("\n" + "=" * 75)
print("CLASS MAPPING")
print("=" * 75)

for encoded_value, class_name in enumerate(class_names):
    print(f"{encoded_value} --> {class_name}")

# ================================================================
# 10. TRAIN-TEST SPLIT
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n" + "=" * 75)
print("TRAIN-TEST SPLIT")
print("=" * 75)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

print(
    "\nTraining percentage:",
    f"{(1 - TEST_SIZE) * 100:.1f}%"
)

print(
    "Testing percentage:",
    f"{TEST_SIZE * 100:.1f}%"
)

# ================================================================
# 11. NUMERICAL PREPROCESSING
# ================================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# ================================================================
# 12. CATEGORICAL PREPROCESSING
# ================================================================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# ================================================================
# 13. COMBINE PREPROCESSING
# ================================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ],
    remainder="drop"
)

# ================================================================
# 14. CREATE LOGISTIC REGRESSION MODEL
# ================================================================

logistic_model = LogisticRegression(
    solver="lbfgs",
    max_iter=2000,
    random_state=RANDOM_STATE
)

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", logistic_model)
    ]
)

# ================================================================
# 15. TRAIN MODEL
# ================================================================

print("\n" + "=" * 75)
print("LOGISTIC REGRESSION")
print("=" * 75)

print("\nTraining model...")

model_pipeline.fit(X_train, y_train)

print("Model training completed successfully.")

# ================================================================
# 16. MAKE PREDICTIONS
# ================================================================

y_pred = model_pipeline.predict(X_test)

print("\nPredictions completed.")

# ================================================================
# 17. CALCULATE CLASS PROBABILITIES
# ================================================================

y_prob = model_pipeline.predict_proba(X_test)

print("\nClass probability calculation completed.")

# ================================================================
# 18. MODEL METRICS
# ================================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test, y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test, y_pred,
    average="weighted",
    zero_division=0
)

cross_entropy = log_loss(
    y_test,
    y_prob,
    labels=np.arange(len(class_names))
)

print("\n" + "=" * 75)
print("MODEL RESULTS")
print("=" * 75)

print(f"\nAccuracy             : {accuracy:.4f}")
print(f"Accuracy (%)         : {accuracy * 100:.2f}%")
print(f"Precision (Weighted) : {precision:.4f}")
print(f"Recall (Weighted)    : {recall:.4f}")
print(f"F1 Score (Weighted)  : {f1:.4f}")
print(f"Cross-Entropy Loss   : {cross_entropy:.4f}")

# ================================================================
# 19. SAVE METRICS
# ================================================================

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Accuracy (%)",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1 Score",
        "Cross-Entropy Loss"
    ],
    "Value": [
        accuracy,
        accuracy * 100,
        precision,
        recall,
        f1,
        cross_entropy
    ]
})

metrics.to_csv(
    os.path.join(OUTPUT_DIR, "Model_Metrics.csv"),
    index=False
)

# ================================================================
# 20. CLASSIFICATION REPORT
# ================================================================

report = classification_report(
    y_test,
    y_pred,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    zero_division=0
)

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

print(report)

with open(
    os.path.join(OUTPUT_DIR, "Classification_Report.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

# ================================================================
# 21. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(class_names))
)

pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
).to_csv(
    os.path.join(OUTPUT_DIR, "Confusion_Matrix.csv")
)

fig, ax = plt.subplots(figsize=(7, 5))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
).plot(
    ax=ax,
    values_format="d",
    colorbar=False
)

ax.set_title("Logistic Regression - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "Confusion_Matrix.png"),
    dpi=300
)

plt.close()

# ================================================================
# 22. SAVE TEST SET PREDICTIONS
# ================================================================

prediction_data = X_test.reset_index(drop=True).copy()

prediction_data["Actual_Attrition"] = (
    label_encoder.inverse_transform(y_test)
)

prediction_data["Predicted_Attrition"] = (
    label_encoder.inverse_transform(y_pred)
)

prediction_data["Correct_Prediction"] = (
    y_test == y_pred
)

for index, class_name in enumerate(class_names):
    safe_name = "".join(
        char if char.isalnum() else "_"
        for char in str(class_name)
    )

    prediction_data[f"Probability_{safe_name}"] = y_prob[:, index]

prediction_data.to_csv(
    os.path.join(OUTPUT_DIR, "Test_Set_Predictions.csv"),
    index=False
)

# ================================================================
# 23. SAVE CLASS PROBABILITIES
# ================================================================

probability_data = pd.DataFrame(
    y_prob,
    columns=[
        f"Probability_{name}"
        for name in class_names
    ]
)

probability_data.insert(
    0,
    "Predicted_Class",
    label_encoder.inverse_transform(y_pred)
)

probability_data.insert(
    0,
    "Actual_Class",
    label_encoder.inverse_transform(y_test)
)

probability_data.to_csv(
    os.path.join(OUTPUT_DIR, "Class_Probabilities.csv"),
    index=False
)

# ================================================================
# 24. PLOT CLASS DISTRIBUTION
# ================================================================

class_counts = (
    data[TARGET_COLUMN]
    .astype(str)
    .value_counts()
    .reindex(class_names, fill_value=0)
)

fig, ax = plt.subplots(figsize=(7, 5))

class_counts.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Target Class Distribution")
ax.set_xlabel("Attrition Class")
ax.set_ylabel("Number of Employees")
ax.tick_params(axis="x", rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "Target_Class_Distribution.png"),
    dpi=300
)

plt.close()

# ================================================================
# 25. SAVE COEFFICIENTS - CORRECTED
# ================================================================

# IMPORTANT:
# Get the fitted model and fitted preprocessor from the pipeline.
# Do not use an undefined variable called "classifier".

fitted_classifier = model_pipeline.named_steps["classifier"]

fitted_preprocessor = model_pipeline.named_steps["preprocessor"]

feature_names = fitted_preprocessor.get_feature_names_out()

# Binary Logistic Regression has one coefficient row.
# The number of rows is taken directly from coef_.
coefficient_values = fitted_classifier.coef_

coefficients = pd.DataFrame(
    coefficient_values,
    columns=feature_names
)

coefficients.insert(
    0,
    "Coefficient_Row",
    [
        f"Coefficient_{i}"
        for i in range(coefficient_values.shape[0])
    ]
)

coefficients.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Logistic_Regression_Coefficients.csv"
    ),
    index=False
)

print("\nCoefficients saved successfully.")

# ================================================================
# 26. SAVE INTERCEPT
# ================================================================

intercepts = pd.DataFrame({
    "Coefficient_Row": [
        f"Coefficient_{i}"
        for i in range(len(fitted_classifier.intercept_))
    ],
    "Intercept": fitted_classifier.intercept_
})

intercepts.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "Logistic_Regression_Intercept.csv"
    ),
    index=False
)

print("Intercept saved successfully.")

# ================================================================
# 27. SAVE COEFFICIENT SUMMARY
# ================================================================

# Absolute coefficient values provide a rough comparison of
# coefficient magnitudes; they are not tree-based feature importance.

if coefficient_values.shape[0] == 1:
    summary_values = coefficient_values[0]
else:
    summary_values = np.mean(
        np.abs(coefficient_values),
        axis=0
    )

coefficient_summary = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": summary_values,
    "Absolute_Coefficient": np.abs(summary_values)
})

coefficient_summary = coefficient_summary.sort_values(
    "Absolute_Coefficient",
    ascending=False
)

coefficient_summary.to_csv(
    os.path.join(OUTPUT_DIR, "Coefficient_Summary.csv"),
    index=False
)

# ================================================================
# 28. PLOT TOP COEFFICIENTS
# ================================================================

top_features = (
    coefficient_summary
    .head(15)
    .sort_values("Absolute_Coefficient", ascending=True)
)

fig, ax = plt.subplots(figsize=(10, 8))

ax.barh(
    top_features["Feature"],
    top_features["Absolute_Coefficient"]
)

ax.set_title("Top 15 Features by Absolute Coefficient")
ax.set_xlabel("Absolute Coefficient")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "Top_Feature_Coefficients.png"),
    dpi=300
)

plt.close()

# ================================================================
# 29. SAVE TRAINED MODEL AND ENCODER
# ================================================================

joblib.dump(
    model_pipeline,
    os.path.join(
        OUTPUT_DIR,
        "Logistic_Regression_Model.pkl"
    )
)

joblib.dump(
    label_encoder,
    os.path.join(
        OUTPUT_DIR,
        "Target_Label_Encoder.pkl"
    )
)

# ================================================================
# 30. SAVE CONFIGURATION
# ================================================================

configuration = pd.DataFrame({
    "Parameter": [
        "Target column",
        "Test size",
        "Random state",
        "Solver",
        "Maximum iterations",
        "Training samples",
        "Testing samples",
        "Original feature count",
        "Encoded feature count",
        "Class names"
    ],
    "Value": [
        TARGET_COLUMN,
        TEST_SIZE,
        RANDOM_STATE,
        "lbfgs",
        2000,
        len(X_train),
        len(X_test),
        X.shape[1],
        len(feature_names),
        ", ".join(class_names)
    ]
})

configuration.to_csv(
    os.path.join(OUTPUT_DIR, "Model_Configuration.csv"),
    index=False
)

# ================================================================
# 31. FINAL SUMMARY
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
    "\nReminder: the existing dataset predicts employee attrition "
    "(Yes/No), not resume-to-job skill matching or placement."
)
_