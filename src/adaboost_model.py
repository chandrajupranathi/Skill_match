
# ================================================================
# ADABOOST CLASSIFIER
# SKILL MATCH PROJECT
# ================================================================
#
# INPUT:
#   dataset/final_preprocessed.csv
#
# OUTPUTS:
#   1. Accuracy, Precision, Recall, F1 Score
#   2. Classification Report
#   3. Confusion Matrix (CSV + PNG)
#   4. Feature Importance (CSV + PNG)
#   5. Actual vs Predicted Results
#   6. Trained Model
#   7. Model Parameters
#   8. Dataset Information
#
# ================================================================

import os
import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")


# ================================================================
# 1. PROJECT PATHS
# ================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR, "dataset", "final_preprocessed.csv"
)

OUTPUT_FOLDER = os.path.join(
    BASE_DIR, "output", "AdaBoost_Classifier_M3_Outputs"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ================================================================
# 2. TARGET COLUMN
# ================================================================

# Change this if your Skill Match dataset uses another target.
TARGET_COLUMN = "Attrition"


# ================================================================
# 3. LOAD DATASET
# ================================================================

print("\n" + "=" * 60)
print("ADABOOST CLASSIFIER - SKILL MATCH")
print("=" * 60)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Dataset not found:\n{INPUT_FILE}\n"
        "Check the dataset filename and project folder."
    )

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))


# ================================================================
# 4. CHECK TARGET COLUMN
# ================================================================

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' was not found.\n"
        f"Available columns: {list(df.columns)}\n"
        "Set TARGET_COLUMN to the correct prediction column."
    )

df = df.dropna(subset=[TARGET_COLUMN]).copy()

if df.empty:
    raise ValueError("No rows remain after removing missing target values.")


# ================================================================
# 5. SEPARATE FEATURES AND TARGET
# ================================================================

X = df.drop(columns=[TARGET_COLUMN]).copy()
y = df[TARGET_COLUMN].copy()

# Remove columns that contain no useful values.
X = X.dropna(axis=1, how="all")

# Remove constant columns.
X = X.loc[:, X.nunique(dropna=False) > 1]

if X.shape[1] == 0:
    raise ValueError("No usable feature columns found.")

if y.nunique() < 2:
    raise ValueError("The target must contain at least two classes.")

print("\nNumber of features:", X.shape[1])
print("Target column:", TARGET_COLUMN)
print("\nTarget distribution:")
print(y.value_counts())


# ================================================================
# 6. ENCODE TARGET
# ================================================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.astype(str))

class_names = label_encoder.classes_
number_of_classes = len(class_names)

print("\nTarget encoding:")
for index, class_name in enumerate(class_names):
    print(f"{class_name} -> {index}")


# ================================================================
# 7. ENCODE CATEGORICAL FEATURES
# ================================================================

# Convert categorical columns to numeric category codes.
# Missing categories are assigned -1.

for column in X.columns:

    if (
        pd.api.types.is_object_dtype(X[column])
        or isinstance(X[column].dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(X[column])
    ):
        values = X[column].astype("string").fillna("__MISSING__")
        X[column] = pd.factorize(values, sort=True)[0]

# Convert boolean columns to integers.
for column in X.columns:
    if pd.api.types.is_bool_dtype(X[column]):
        X[column] = X[column].astype(int)

# Convert all remaining columns to numeric.
X = X.apply(pd.to_numeric, errors="coerce")

# Replace infinite values with missing values.
X = X.replace([np.inf, -np.inf], np.nan)


# ================================================================
# 8. HANDLE MISSING FEATURE VALUES
# ================================================================

imputer = SimpleImputer(strategy="median")

X_imputed = imputer.fit_transform(X)

# Keep feature names for reports.
feature_names = X.columns.tolist()

X = pd.DataFrame(
    X_imputed,
    columns=feature_names,
    index=df.index
)

print("\nMissing feature values handled successfully.")


# ================================================================
# 9. TRAIN-TEST SPLIT
# ================================================================

class_counts = pd.Series(y_encoded).value_counts()

if class_counts.min() < 2:
    raise ValueError(
        "Each target class needs at least 2 rows for a stratified split."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ================================================================
# 10. CREATE ADABOOST MODEL
# ================================================================

base_model = DecisionTreeClassifier(
    max_depth=1,
    random_state=42
)

model = AdaBoostClassifier(
    estimator=base_model,
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

print("\nAdaBoost model created successfully.")


# ================================================================
# 11. TRAIN MODEL
# ================================================================

model.fit(X_train, y_train)

print("Model training completed!")


# ================================================================
# 12. MAKE PREDICTIONS
# ================================================================

y_pred = model.predict(X_test)

print("\nPredictions completed!")


# ================================================================
# 13. CALCULATE EVALUATION METRICS
# ================================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, average="weighted", zero_division=0
)

recall = recall_score(
    y_test, y_pred, average="weighted", zero_division=0
)

f1 = f1_score(
    y_test, y_pred, average="weighted", zero_division=0
)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ================================================================
# 14. SAVE METRICS
# ================================================================

metrics = {
    "Model": "AdaBoost Classifier",
    "Target Column": TARGET_COLUMN,
    "Accuracy": float(accuracy),
    "Precision": float(precision),
    "Recall": float(recall),
    "F1 Score": float(f1),
    "Training Samples": int(len(X_train)),
    "Testing Samples": int(len(X_test)),
    "Number of Features": int(X.shape[1]),
    "Number of Classes": int(number_of_classes)
}

metrics_df = pd.DataFrame([metrics])

metrics_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "evaluation_metrics.csv"),
    index=False
)

with open(
    os.path.join(OUTPUT_FOLDER, "evaluation_metrics.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=4)


# ================================================================
# 15. CLASSIFICATION REPORT
# ================================================================

report = classification_report(
    y_test,
    y_pred,
    labels=np.arange(number_of_classes),
    target_names=[str(name) for name in class_names],
    zero_division=0
)

print("\nClassification Report:\n")
print(report)

with open(
    os.path.join(OUTPUT_FOLDER, "classification_report.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write(report)


# ================================================================
# 16. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(number_of_classes)
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual_{name}" for name in class_names],
    columns=[f"Predicted_{name}" for name in class_names]
)

cm_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "confusion_matrix.csv")
)

fig, ax = plt.subplots(figsize=(8, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("AdaBoost Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "confusion_matrix.png"),
    dpi=300
)

plt.close(fig)


# ================================================================
# 17. FEATURE IMPORTANCE
# ================================================================

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "feature_importance.csv"),
    index=False
)

top_features = importance_df.head(20).sort_values(
    by="Importance",
    ascending=True
)

fig, ax = plt.subplots(figsize=(10, 7))

ax.barh(
    top_features["Feature"],
    top_features["Importance"]
)

ax.set_title("AdaBoost - Top 20 Feature Importances")
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "feature_importance.png"),
    dpi=300
)

plt.close(fig)


# ================================================================
# 18. ACTUAL VS PREDICTED RESULTS
# ================================================================

prediction_df = pd.DataFrame({
    "Actual_Encoded": y_test,
    "Predicted_Encoded": y_pred,
    "Actual_Label": label_encoder.inverse_transform(y_test),
    "Predicted_Label": label_encoder.inverse_transform(y_pred),
    "Correct_Prediction": y_test == y_pred
})

prediction_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "predictions.csv"),
    index=False
)


# ================================================================
# 19. SAVE TRAINED MODEL
# ================================================================

model_path = os.path.join(
    OUTPUT_FOLDER,
    "adaboost_model.pkl"
)

joblib.dump(
    {
        "model": model,
        "imputer": imputer,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "target_column": TARGET_COLUMN
    },
    model_path
)

print("\nTrained model saved successfully!")


# ================================================================
# 20. SAVE MODEL PARAMETERS
# ================================================================

parameters = model.get_params()

with open(
    os.path.join(OUTPUT_FOLDER, "model_parameters.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        {key: str(value) for key, value in parameters.items()},
        file,
        indent=4
    )


# ================================================================
# 21. SAVE DATASET INFORMATION
# ================================================================

dataset_info = {
    "Dataset File": os.path.basename(INPUT_FILE),
    "Original Rows": int(len(df)),
    "Feature Count": int(X.shape[1]),
    "Target Column": TARGET_COLUMN,
    "Classes": [str(name) for name in class_names],
    "Training Rows": int(len(X_train)),
    "Testing Rows": int(len(X_test)),
    "Missing Target Rows Removed": int(
        pd.read_csv(INPUT_FILE)[TARGET_COLUMN].isna().sum()
    )
}

with open(
    os.path.join(OUTPUT_FOLDER, "dataset_info.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump(dataset_info, file, indent=4)


# ================================================================
# 22. FINAL OUTPUT
# ================================================================

print("\n" + "=" * 60)
print("ADABOOST EXECUTION COMPLETED")
print("=" * 60)

print("\nOutput folder:")
print(OUTPUT_FOLDER)

print("\nGenerated files:")
for filename in sorted(os.listdir(OUTPUT_FOLDER)):
    print(" -", filename)

print("\nAll results saved successfully!")
