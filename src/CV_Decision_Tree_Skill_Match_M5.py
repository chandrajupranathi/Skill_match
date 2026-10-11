
# ============================================================
# DECISION TREE CLASSIFIER WITH CROSS VALIDATION
# SKILL MATCH - EMPLOYEE ATTRITION DATASET
#
# IMPORTANT:
# 1. Original dataset is NOT modified.
# 2. All outputs are stored in ONE folder.
# 3. Uses Decision Tree Classifier only.
# 4. Uses 5-Fold Stratified Cross Validation.
# 5. Preprocessing is performed inside a Pipeline to prevent
#    data leakage between cross-validation folds.
#
# OUTPUTS:
# 1. Accuracy, Precision, Recall and F1 Score
# 2. ROC-AUC and PR-AUC
# 3. Confusion Matrix CSV and Graph
# 4. Classification Report
# 5. Cross-Validated Predictions
# 6. ROC and Precision-Recall Curves
# 7. Feature Importance CSV and Graph
# 8. Dataset Information
# 9. Model Configuration
# 10. Trained Model
# ============================================================

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT AND PATH SETTINGS
# ============================================================

PROJECT_PATH = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# The script checks common raw and cleaned employee datasets.
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
        "dataset",
        "reordered_raw.csv"
    ),
    os.path.join(
        PROJECT_PATH,
        "dataset",
        "cleaned_mean_model_imputed.csv"
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

# All outputs go into the existing "output" directory.
OUTPUT_FOLDER = os.path.join(
    PROJECT_PATH,
    "output",
    "CV_Decision_Tree_Skill_Match_M5_Outputs"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 75)
print("DECISION TREE CLASSIFIER - SKILL MATCH")
print("EMPLOYEE ATTRITION PREDICTION")
print("=" * 75)

if INPUT_FILE is None:
    print("\nDataset not found. Checked these locations:")

    for path in DATASET_CANDIDATES:
        print(" -", path)

    raise FileNotFoundError(
        "Place the correct employee attrition CSV in the dataset "
        "folder or update DATASET_CANDIDATES."
    )

print("\nInput dataset:", INPUT_FILE)

data = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")
print("Original dataset shape:", data.shape)

if data.empty:
    raise ValueError("The dataset is empty.")

# Work on a copy; never overwrite the source CSV.
working_data = data.copy()

working_data.columns = (
    working_data.columns.astype(str).str.strip()
)


# ============================================================
# 3. REMOVE DUPLICATES FROM WORKING COPY
# ============================================================

duplicate_count = int(working_data.duplicated().sum())

if duplicate_count > 0:
    working_data = (
        working_data.drop_duplicates()
        .reset_index(drop=True)
    )

print("\nDuplicate rows found:", duplicate_count)
print("Working dataset shape:", working_data.shape)


# ============================================================
# 4. DETECT ATTRITION TARGET
# ============================================================

target_candidates = [
    "Attrition",
    "attrition",
    "ATTRITION"
]

target_column = next(
    (
        column for column in target_candidates
        if column in working_data.columns
    ),
    None
)

if target_column is None:
    raise ValueError(
        "Attrition target was not found. Available columns:\n"
        + str(list(working_data.columns))
        + "\nPlease check the input dataset."
    )

print("\nTarget column:", target_column)

# Remove rows with missing target values.
working_data = working_data.dropna(
    subset=[target_column]
).reset_index(drop=True)

if working_data.empty:
    raise ValueError("No rows remain after removing missing targets.")


# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

X = working_data.drop(
    columns=[target_column]
).copy()

y = working_data[target_column].astype(str).str.strip()

# Remove accidental blank targets.
valid_target = y.ne("")

X = X.loc[valid_target].reset_index(drop=True)
y = y.loc[valid_target].reset_index(drop=True)

if len(y) == 0:
    raise ValueError("No valid target labels remain.")


# ============================================================
# 6. REMOVE IDENTIFIER AND CONSTANT FEATURES
# ============================================================

# EmployeeNumber is an identifier in the common IBM dataset.
# It should not be used as a predictive feature.
identifier_candidates = [
    "EmployeeNumber",
    "EmployeeID",
    "Employee_ID",
    "Record_ID"
]

identifier_columns = [
    column for column in X.columns
    if column.lower() in {
        name.lower() for name in identifier_candidates
    }
]

if identifier_columns:
    print("\nIdentifier columns excluded:", identifier_columns)
    X = X.drop(columns=identifier_columns)

constant_columns = [
    column for column in X.columns
    if X[column].nunique(dropna=False) <= 1
]

if constant_columns:
    print("Constant columns excluded:", constant_columns)
    X = X.drop(columns=constant_columns)

if X.shape[1] == 0:
    raise ValueError("No usable features remain.")


# ============================================================
# 7. ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

class_labels = label_encoder.classes_
number_of_classes = len(class_labels)

print("\nTarget classes:")

for index, label in enumerate(class_labels):
    print(index, "=", label)

if number_of_classes < 2:
    raise ValueError("Attrition must contain at least two classes.")

class_counts = pd.Series(y_encoded).value_counts()

if class_counts.min() < 5:
    raise ValueError(
        "Five-fold stratified cross-validation requires at least "
        "five records in every target class."
    )

print("\nNumber of classes:", number_of_classes)


# ============================================================
# 8. IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool", "string"]
).columns.tolist()

print("\nNumeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


# ============================================================
# 9. NUMERIC PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# ============================================================
# 10. CATEGORICAL PREPROCESSING
# ============================================================

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


# ============================================================
# 11. COLUMN TRANSFORMER
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ],
    remainder="drop"
)


# ============================================================
# 12. DECISION TREE MODEL AND PIPELINE
# ============================================================

decision_tree = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", decision_tree)
    ]
)

print("\nDecision Tree pipeline created.")
print("Criterion: entropy")


# ============================================================
# 13. STRATIFIED K-FOLD CROSS VALIDATION
# ============================================================

N_SPLITS = 5

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=42
)

print(
    "\nPerforming",
    N_SPLITS,
    "-Fold Stratified Cross Validation..."
)


# ============================================================
# 14. CROSS-VALIDATED CLASS PREDICTIONS
# ============================================================

y_pred = cross_val_predict(
    model,
    X,
    y_encoded,
    cv=cv,
    method="predict",
    n_jobs=-1
)

print("Cross-validated predictions completed.")


# ============================================================
# 15. CROSS-VALIDATED PROBABILITY PREDICTIONS
# ============================================================

y_probability = cross_val_predict(
    model,
    X,
    y_encoded,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)

print("Probability predictions completed.")


# ============================================================
# 16. CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(y_encoded, y_pred)

precision = precision_score(
    y_encoded,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_encoded,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_encoded,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 75)
print("CROSS-VALIDATED PERFORMANCE")
print("=" * 75)

print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_encoded,
    y_pred,
    labels=np.arange(number_of_classes)
)

print("\nConfusion Matrix:")
print(cm)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual_{c}" for c in class_labels],
    columns=[f"Predicted_{c}" for c in class_labels]
)

cm_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "confusion_matrix.csv")
)


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_encoded,
    y_pred,
    labels=np.arange(number_of_classes),
    target_names=[str(c) for c in class_labels],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    os.path.join(OUTPUT_FOLDER, "classification_report.csv")
)

print("\nClassification Report:")
print(
    classification_report(
        y_encoded,
        y_pred,
        labels=np.arange(number_of_classes),
        target_names=[str(c) for c in class_labels],
        zero_division=0
    )
)


# ============================================================
# 19. ROC-AUC
# ============================================================

roc_auc = np.nan

try:
    if number_of_classes == 2:
        roc_auc = roc_auc_score(
            y_encoded,
            y_probability[:, 1]
        )

        fpr, tpr, _ = roc_curve(
            y_encoded,
            y_probability[:, 1]
        )

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            label=f"ROC-AUC = {roc_auc:.4f}"
        )
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Skill Match - Decision Tree ROC Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            os.path.join(OUTPUT_FOLDER, "ROC_AUC_Curve.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    else:
        roc_auc = roc_auc_score(
            y_encoded,
            y_probability,
            multi_class="ovr",
            average="weighted",
            labels=np.arange(number_of_classes)
        )

except Exception as error:
    print("\nROC-AUC could not be calculated:", error)


# ============================================================
# 20. PRECISION-RECALL AUC
# ============================================================

pr_auc = np.nan

try:
    if number_of_classes == 2:
        pr_auc = average_precision_score(
            y_encoded,
            y_probability[:, 1]
        )

        precision_curve, recall_curve, _ = (
            precision_recall_curve(
                y_encoded,
                y_probability[:, 1]
            )
        )

        plt.figure(figsize=(8, 6))
        plt.plot(
            recall_curve,
            precision_curve,
            label=f"Average Precision = {pr_auc:.4f}"
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Skill Match - Precision-Recall Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            os.path.join(OUTPUT_FOLDER, "PR_AUC_Curve.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    else:
        pr_auc = average_precision_score(
            np.eye(number_of_classes)[y_encoded],
            y_probability,
            average="weighted"
        )

except Exception as error:
    print("\nPR-AUC could not be calculated:", error)


# ============================================================
# 21. SAVE CROSS-VALIDATED PREDICTIONS
# ============================================================

prediction_output = working_data.copy()

prediction_output["Actual_Encoded"] = y_encoded
prediction_output["Predicted_Encoded"] = y_pred

prediction_output["Actual_Label"] = (
    label_encoder.inverse_transform(y_encoded)
)

prediction_output["Predicted_Label"] = (
    label_encoder.inverse_transform(y_pred)
)

prediction_output["Correct_Prediction"] = (
    y_encoded == y_pred
)

# Save probability for each class.
for index, label in enumerate(class_labels):
    prediction_output[
        f"Probability_{label}"
    ] = y_probability[:, index]

prediction_output.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "cross_validated_predictions.csv"
    ),
    index=False
)


# ============================================================
# 22. FIT FINAL MODEL FOR FEATURE IMPORTANCE
# ============================================================

print("\nTraining final Decision Tree for model export...")

model.fit(X, y_encoded)

fitted_preprocessor = model.named_steps["preprocessor"]
fitted_classifier = model.named_steps["classifier"]

feature_names = fitted_preprocessor.get_feature_names_out()

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": fitted_classifier.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

feature_importance.to_csv(
    os.path.join(OUTPUT_FOLDER, "feature_importance.csv"),
    index=False
)


# ============================================================
# 23. FEATURE IMPORTANCE GRAPH
# ============================================================

top_n = min(20, len(feature_importance))

top_features = feature_importance.head(top_n).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(11, 8))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Employee Feature")
plt.title("Skill Match - Top Decision Tree Features")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "feature_importance.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 24. CONFUSION MATRIX GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.imshow(cm, interpolation="nearest")
plt.title("Skill Match - Decision Tree Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    range(number_of_classes),
    class_labels,
    rotation=45
)

plt.yticks(
    range(number_of_classes),
    class_labels
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_FOLDER, "confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 25. SAVE PERFORMANCE RESULTS
# ============================================================

performance_results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision_Weighted",
        "Recall_Weighted",
        "F1_Weighted",
        "ROC_AUC",
        "PR_AUC"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        pr_auc
    ]
})

performance_results["Percentage"] = (
    performance_results["Score"] * 100
)

performance_results.to_csv(
    os.path.join(OUTPUT_FOLDER, "performance_results.csv"),
    index=False
)


# ============================================================
# 26. SAVE DATASET INFORMATION
# ============================================================

dataset_info = pd.DataFrame({
    "Parameter": [
        "Input Dataset",
        "Original Rows",
        "Original Columns",
        "Working Rows",
        "Working Columns",
        "Target Column",
        "Number of Classes",
        "Original Feature Count",
        "Numeric Features",
        "Categorical Features",
        "Identifier Columns Removed",
        "Constant Columns Removed",
        "Duplicate Rows Removed",
        "Cross Validation Folds"
    ],
    "Value": [
        INPUT_FILE,
        data.shape[0],
        data.shape[1],
        working_data.shape[0],
        working_data.shape[1],
        target_column,
        number_of_classes,
        X.shape[1],
        len(numeric_features),
        len(categorical_features),
        ", ".join(identifier_columns),
        ", ".join(constant_columns),
        duplicate_count,
        N_SPLITS
    ]
})

dataset_info.to_csv(
    os.path.join(OUTPUT_FOLDER, "dataset_information.csv"),
    index=False
)


# ============================================================
# 27. SAVE MODEL CONFIGURATION
# ============================================================

model_configuration = pd.DataFrame({
    "Parameter": [
        "Algorithm",
        "Criterion",
        "Maximum Depth",
        "Minimum Samples Split",
        "Minimum Samples Leaf",
        "Cross Validation",
        "Number of Folds",
        "Shuffle",
        "Random State",
        "Numeric Imputation",
        "Categorical Imputation",
        "Categorical Encoding",
        "Target Column"
    ],
    "Value": [
        "Decision Tree Classifier",
        "entropy",
        "None",
        2,
        1,
        "Stratified K-Fold",
        N_SPLITS,
        True,
        42,
        "Median",
        "Most Frequent",
        "One-Hot Encoding",
        target_column
    ]
})

model_configuration.to_csv(
    os.path.join(OUTPUT_FOLDER, "model_configuration.csv"),
    index=False
)


# ============================================================
# 28. SAVE TRAINED MODEL AND LABEL ENCODER
# ============================================================

with open(
    os.path.join(OUTPUT_FOLDER, "decision_tree_model.pkl"),
    "wb"
) as file:
    pickle.dump(model, file)

with open(
    os.path.join(OUTPUT_FOLDER, "target_label_encoder.pkl"),
    "wb"
) as file:
    pickle.dump(label_encoder, file)


# ============================================================
# 29. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("SKILL MATCH DECISION TREE COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nCross-validated performance:")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")

print("\nOriginal dataset was NOT modified.")
print("\nDataset used:")
print(INPUT_FILE)

print("\nAll outputs saved to:")
print(OUTPUT_FOLDER)

print("\nGenerated files:")

for filename in sorted(os.listdir(OUTPUT_FOLDER)):
    print(" -", filename)

print("\n" + "=" * 75)
print("PROGRAM FINISHED")
print("=" * 75)
