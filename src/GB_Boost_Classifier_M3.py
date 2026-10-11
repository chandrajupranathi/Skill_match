
# ================================================================
# GRADIENT BOOSTING CLASSIFIER
# SKILL MATCH PROJECT
# ================================================================
#
# IMPORTANT:
# The original dataset is NEVER modified.
#
# OUTPUTS:
#   1. Accuracy, Precision, Recall, F1 Score
#   2. Classification Report
#   3. Predictions CSV
#   4. Confusion Matrix CSV and PNG
#   5. Feature Importance CSV and PNG
#   6. ROC Curve
#   7. Model Parameters
#   8. Dataset Information
#
# ================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

warnings.filterwarnings("ignore")

# ================================================================
# 1. PATH CONFIGURATION
# ================================================================

DATA_PATH = (
    r"C:\Users\chand\Downloads\Skill_Match (2)"
    r"\Skill_Match\dataset\reordered_raw.csv"
)

OUTPUT_DIR = (
    r"C:\Users\chand\Downloads\Skill_Match (2)"
    r"\Skill_Match\output\GB_Boost_Classifier_M3_Outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# 2. LOAD DATASET
# ================================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("GRADIENT BOOSTING CLASSIFIER - SKILL MATCH")
print("=" * 65)

print("\nDataset Shape:", df.shape)
print("\nDataset Columns:")
print(df.columns.tolist())

# ================================================================
# 3. IDENTIFY TARGET COLUMN
# ================================================================

# These are possible target names for a Skill Match dataset.
# The target must represent the result you want to predict.

target_candidates = [
    "SkillMatch",
    "Skill_Match",
    "MatchStatus",
    "Match_Status",
    "Match",
    "IsMatch",
    "is_match",
    "MatchResult",
    "Match_Result",
    "Recommended",
    "PlacementStatus",
    "Placement_Status",
    "Placement",
    "Placed",
    "Status",
    "Target",
    "Attrition"
]

target_column = next(
    (
        col for col in target_candidates
        if col in df.columns
    ),
    None
)

if target_column is None:
    raise ValueError(
        "Target column was not identified automatically.\n"
        "Set target_column to the correct prediction column "
        "in your dataset."
    )

print("\nSelected Target Column:", target_column)

# Remove rows with missing target values.
df = df.dropna(subset=[target_column]).copy()

if df.empty:
    raise ValueError("No rows remain after removing missing targets.")

# ================================================================
# 4. SEPARATE FEATURES AND TARGET
# ================================================================

X = df.drop(columns=[target_column]).copy()
y = df[target_column].copy()

# Remove columns that cannot provide useful feature information.
# This avoids accidental learning from constant columns.

constant_columns = [
    col for col in X.columns
    if X[col].nunique(dropna=False) <= 1
]

if constant_columns:
    X = X.drop(columns=constant_columns)

print("\nConstant Columns Removed:", constant_columns)
print("Number of Features:", X.shape[1])

# Encode the target labels.
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y.astype(str))

class_counts = pd.Series(y).value_counts()

if len(class_counts) < 2:
    raise ValueError(
        "The selected target has fewer than two classes. "
        "Choose a suitable prediction target."
    )

# ================================================================
# 5. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ================================================================

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNumerical Features:", len(numeric_features))
print("Categorical Features:", len(categorical_features))

# ================================================================
# 6. DATA PREPROCESSING
# ================================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ],
    remainder="drop"
)

# ================================================================
# 7. TRAIN-TEST SPLIT
# ================================================================

# Use stratification only when every class has enough samples.
stratify_target = (
    y if class_counts.min() >= 2 else None
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=stratify_target
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ================================================================
# 8. GRADIENT BOOSTING MODEL
# ================================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.10,
                max_depth=3,
                random_state=42
            )
        )
    ]
)

# ================================================================
# 9. TRAIN MODEL
# ================================================================

print("\nTraining Gradient Boosting Model...")

model.fit(X_train, y_train)

print("Model Training Completed Successfully!")

# ================================================================
# 10. PREDICTIONS
# ================================================================

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)

# ================================================================
# 11. EVALUATION METRICS
# ================================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

metrics_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "gradient_boosting_metrics.csv"
    ),
    index=False
)

print("\nEvaluation Metrics")
print(metrics_df.to_string(index=False))

# ================================================================
# 12. CLASSIFICATION REPORT
# ================================================================

report = classification_report(
    y_test,
    y_pred,
    labels=np.arange(len(label_encoder.classes_)),
    target_names=[
        str(label)
        for label in label_encoder.classes_
    ],
    zero_division=0
)

print("\nClassification Report:\n")
print(report)

with open(
    os.path.join(OUTPUT_DIR, "classification_report.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

# ================================================================
# 13. SAVE PREDICTIONS
# ================================================================

predictions_df = pd.DataFrame({
    "Actual": label_encoder.inverse_transform(y_test),
    "Predicted": label_encoder.inverse_transform(y_pred),
    "Correct": y_test == y_pred
})

for i, class_name in enumerate(label_encoder.classes_):
    predictions_df[
        f"Probability_{class_name}"
    ] = y_probability[:, i]

predictions_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "gradient_boosting_predictions.csv"
    ),
    index=False
)

# ================================================================
# 14. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(label_encoder.classes_))
)

cm_df = pd.DataFrame(
    cm,
    index=[
        f"Actual_{label}"
        for label in label_encoder.classes_
    ],
    columns=[
        f"Predicted_{label}"
        for label in label_encoder.classes_
    ]
)

cm_df.to_csv(
    os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Gradient Boosting - Skill Match Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    dpi=300
)
plt.close()

# ================================================================
# 15. FEATURE IMPORTANCE
# ================================================================

preprocessor_fitted = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

feature_names = preprocessor_fitted.get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": classifier.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

importance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "gradient_boosting_feature_importance.csv"
    ),
    index=False
)

top_features = importance_df.head(20).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 7))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Gradient Boosting - Top Feature Importance")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "gradient_boosting_feature_importance.png"
    ),
    dpi=300
)
plt.close()

# ================================================================
# 16. ROC CURVE
# ================================================================

# ROC is calculated for binary classification.
if len(label_encoder.classes_) == 2:
    positive_class = 1

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_probability[:, positive_class],
        pos_label=positive_class
    )

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr,
        tpr,
        label=f"ROC Curve (AUC = {roc_auc:.3f})"
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Gradient Boosting - ROC Curve")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "gradient_boosting_roc_curve.png"
        ),
        dpi=300
    )
    plt.close()

# ================================================================
# 17. SAVE MODEL PARAMETERS
# ================================================================

parameters_df = pd.DataFrame({
    "Parameter": [
        "Algorithm",
        "n_estimators",
        "learning_rate",
        "max_depth",
        "test_size",
        "random_state",
        "target_column"
    ],
    "Value": [
        "Gradient Boosting Classifier",
        100,
        0.10,
        3,
        0.20,
        42,
        target_column
    ]
})

parameters_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "gradient_boosting_parameters.csv"
    ),
    index=False
)

# ================================================================
# 18. DATASET INFORMATION
# ================================================================

dataset_info = pd.DataFrame({
    "Property": [
        "Original Dataset Path",
        "Target Column",
        "Rows Used",
        "Original Feature Count",
        "Features After Constant Removal",
        "Numerical Features",
        "Categorical Features",
        "Training Samples",
        "Testing Samples",
        "Number of Classes"
    ],
    "Value": [
        DATA_PATH,
        target_column,
        len(df),
        len(df.columns) - 1,
        X.shape[1],
        len(numeric_features),
        len(categorical_features),
        len(X_train),
        len(X_test),
        len(label_encoder.classes_)
    ]
})

dataset_info.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dataset_information.csv"
    ),
    index=False
)

# ================================================================
# 19. FINAL OUTPUT
# ================================================================

print("\n" + "=" * 65)
print("SKILL MATCH - GRADIENT BOOSTING COMPLETED")
print("=" * 65)

print("\nOutput Folder:")
print(OUTPUT_DIR)

print("\nGenerated Files:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("-", filename)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("All results saved successfully!")
