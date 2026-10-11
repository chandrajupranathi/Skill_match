
# ================================================================
# ADABOOST CLASSIFIER
# SKILL MATCH - EMPLOYEE ATTRITION PREDICTION
# ================================================================
#
# IMPORTANT:
# 1. Original RAW dataset is NEVER modified.
# 2. Preprocessing is performed inside a Pipeline.
# 3. Supports Attrition values such as Yes/No or 1/0.
# 4. All outputs are saved in a dedicated output folder.
#
# OUTPUTS:
#   1. Accuracy, Precision, Recall, F1 Score
#   2. Classification Report
#   3. Confusion Matrix CSV and Graph
#   4. Performance Graph
#   5. Actual vs Predicted Chart
#   6. Attrition Class Distribution
#   7. Feature Importance CSV and Graph
#   8. First AdaBoost Base Tree Visualization
#   9. Test Predictions CSV
#  10. Trained Model
#  11. AdaBoost Parameters
#  12. Estimator Weights
#
# ================================================================


# ================================================================
# 1. IMPORT LIBRARIES
# ================================================================

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")


# ================================================================
# 2. PROJECT PATH
# ================================================================
# This script is expected to be inside the project's src folder.
# The project root is the parent folder of src.
# ================================================================

PROJECT_PATH = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# ================================================================
# 3. RAW DATASET PATH
# ================================================================
# The script checks common dataset locations.
# If necessary, add your actual CSV path to this list.
# ================================================================

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

DATASET_PATH = next(
    (
        path for path in DATASET_CANDIDATES
        if os.path.isfile(path)
    ),
    None
)


# ================================================================
# 4. OUTPUT FOLDER
# ================================================================

OUTPUT_FOLDER = os.path.join(
    PROJECT_PATH,
    "output",
    "AdaBoost_Classifier_M3_Outputs"
)
METRICS_FOLDER = os.path.join(
    OUTPUT_FOLDER, "metrics"
)

PREDICTIONS_FOLDER = os.path.join(
    OUTPUT_FOLDER, "predictions"
)

CONFUSION_FOLDER = os.path.join(
    OUTPUT_FOLDER, "confusion_matrix"
)

CHARTS_FOLDER = os.path.join(
    OUTPUT_FOLDER, "charts"
)

TREE_FOLDER = os.path.join(
    OUTPUT_FOLDER, "adaboost_tree"
)

FEATURE_FOLDER = os.path.join(
    OUTPUT_FOLDER, "feature_importance"
)

MODEL_FOLDER = os.path.join(
    OUTPUT_FOLDER, "model"
)


# ================================================================
# 5. CREATE OUTPUT FOLDERS
# ================================================================

folders = [
    OUTPUT_FOLDER,
    METRICS_FOLDER,
    PREDICTIONS_FOLDER,
    CONFUSION_FOLDER,
    CHARTS_FOLDER,
    TREE_FOLDER,
    FEATURE_FOLDER,
    MODEL_FOLDER
]

for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print("Created folder:", folder)
    else:
        print("Folder already exists:", folder)


# ================================================================
# 6. PROGRAM HEADER
# ================================================================

print("\n" + "=" * 75)
print("       SKILL MATCH - ADABOOST EMPLOYEE ATTRITION")
print("=" * 75)


# ================================================================
# 7. CHECK AND LOAD DATASET
# ================================================================

if DATASET_PATH is None:
    print("\nERROR: Employee attrition dataset not found.")
    print("\nExpected one of these locations:")

    for path in DATASET_CANDIDATES:
        print(path)

    raise FileNotFoundError(
        "Place the raw employee attrition CSV in the dataset folder "
        "or update DATASET_CANDIDATES."
    )

print("\nDataset found:", DATASET_PATH)

df = pd.read_csv(DATASET_PATH)

print("\nRaw dataset loaded successfully.")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ================================================================
# 8. CREATE A COPY OF THE DATASET
# ================================================================

data = df.copy()

print("\nDataset columns:")
print(list(data.columns))

print("\nFirst 5 records:")
print(data.head())


# ================================================================
# 9. IDENTIFY TARGET COLUMN
# ================================================================

target_column = "Attrition"

# Handle accidental spaces in column names.
data.columns = data.columns.astype(str).str.strip()

if target_column not in data.columns:
    print("\nERROR: Attrition target column not found.")
    print("Available columns:", list(data.columns))
    raise KeyError(
        "Check the target column name in your CSV."
    )

print("\nTarget column:", target_column)


# ================================================================
# 10. CLEAN TARGET VALUES
# ================================================================

# Only the in-memory copy is changed.
data[target_column] = data[target_column].replace(
    r"^\s*$", np.nan, regex=True
)

data = data.dropna(
    subset=[target_column]
).copy()

# Normalize textual Yes/No labels while preserving numeric 0/1.
if (
    pd.api.types.is_object_dtype(data[target_column])
    or isinstance(data[target_column].dtype, pd.CategoricalDtype)
):
    data[target_column] = (
        data[target_column].astype(str).str.strip()
    )

    data[target_column] = data[target_column].replace({
        "YES": "Yes",
        "NO": "No",
        "yes": "Yes",
        "no": "No"
    })

y = data[target_column].copy()

if y.nunique() != 2:
    raise ValueError(
        "AdaBoost Attrition classifier expects exactly 2 target "
        f"classes. Found: {y.unique().tolist()}"
    )

print("\nAttrition class distribution:")
print(y.value_counts())


# ================================================================
# 11. SEPARATE FEATURES AND TARGET
# ================================================================

X = data.drop(
    columns=[target_column]
).copy()

# Remove identifier columns that should not teach the model
# a person's identity.
identifier_columns = [
    column for column in X.columns
    if column.strip().lower() in [
        "employeenumber",
        "employeeid",
        "employee_id"
    ]
]

if identifier_columns:
    X.drop(
        columns=identifier_columns,
        inplace=True
    )
    print("\nRemoved identifier columns:", identifier_columns)


# ================================================================
# 12. REMOVE COMPLETELY EMPTY AND CONSTANT FEATURES
# ================================================================

empty_columns = X.columns[
    X.isnull().all()
].tolist()

constant_columns = [
    column for column in X.columns
    if X[column].nunique(dropna=True) <= 1
]

columns_to_remove = list(
    dict.fromkeys(empty_columns + constant_columns)
)

if columns_to_remove:
    print("\nRemoved empty/constant features:", columns_to_remove)
    X.drop(
        columns=columns_to_remove,
        inplace=True
    )

if X.shape[1] == 0:
    raise ValueError("No usable feature columns remain.")


# ================================================================
# 13. IDENTIFY NUMERIC AND CATEGORICAL FEATURES
# ================================================================

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Convert uncommon remaining data types to strings so they can
# be handled by the categorical preprocessing pipeline.
remaining_features = [
    column for column in X.columns
    if column not in numeric_features
    and column not in categorical_features
]

for column in remaining_features:
    X[column] = X[column].astype("string")

categorical_features.extend(remaining_features)

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ================================================================
# 14. NUMERIC PREPROCESSING
# ================================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# ================================================================
# 15. CATEGORICAL PREPROCESSING
# ================================================================

try:
    onehot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
except TypeError:
    # Compatibility with older scikit-learn versions.
    onehot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            onehot_encoder
        )
    ]
)


# ================================================================
# 16. COLUMN TRANSFORMER
# ================================================================

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


# ================================================================
# 17. TRAIN-TEST SPLIT
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records :", len(X_test))


# ================================================================
# 18. CREATE BASE DECISION TREE
# ================================================================

base_tree = DecisionTreeClassifier(
    max_depth=2,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)


# ================================================================
# 19. CREATE ADABOOST CLASSIFIER
# ================================================================

try:
    adaboost = AdaBoostClassifier(
        estimator=base_tree,
        n_estimators=100,
        learning_rate=0.8,
        random_state=42
    )
except TypeError:
    # Compatibility with older scikit-learn versions.
    adaboost = AdaBoostClassifier(
        base_estimator=base_tree,
        n_estimators=100,
        learning_rate=0.8,
        random_state=42
    )


# ================================================================
# 20. CREATE COMPLETE PIPELINE
# ================================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", adaboost)
    ]
)


# ================================================================
# 21. TRAIN MODEL
# ================================================================

print("\nTraining AdaBoost Classifier...")

model.fit(
    X_train,
    y_train
)

print("Training completed successfully.")


# ================================================================
# 22. GENERATE PREDICTIONS
# ================================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

print("Predictions generated successfully.")


# ================================================================
# 23. CALCULATE PERFORMANCE METRICS
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


# ================================================================
# 24. DISPLAY PERFORMANCE
# ================================================================

print("\n" + "=" * 75)
print("           ADABOOST ATTRITION PERFORMANCE")
print("=" * 75)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nPerformance Percentage:")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")


# ================================================================
# 25. SAVE METRICS CSV
# ================================================================

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
    ],
    "Percentage": [
        accuracy * 100,
        precision * 100,
        recall * 100,
        f1 * 100
    ]
})

metrics_df.to_csv(
    os.path.join(METRICS_FOLDER, "adaboost_metrics.csv"),
    index=False
)


# ================================================================
# 26. SAVE CLASSIFICATION REPORT
# ================================================================

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    os.path.join(METRICS_FOLDER, "classification_report.csv")
)


# ================================================================
# 27. CONFUSION MATRIX
# ================================================================

adaboost_classifier = model.named_steps["classifier"]
class_labels = adaboost_classifier.classes_

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=class_labels
)

print("\nConfusion Matrix:")
print(cm)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual_{label}" for label in class_labels],
    columns=[f"Predicted_{label}" for label in class_labels]
)

cm_df.to_csv(
    os.path.join(CONFUSION_FOLDER, "confusion_matrix.csv")
)


# ================================================================
# 28. CONFUSION MATRIX GRAPH
# ================================================================

fig, ax = plt.subplots(figsize=(8, 6))

image = ax.imshow(cm, cmap="Blues")

ax.set_title("Skill Match - AdaBoost Confusion Matrix")
ax.set_xlabel("Predicted Attrition")
ax.set_ylabel("Actual Attrition")

ax.set_xticks(range(len(class_labels)))
ax.set_xticklabels(class_labels, rotation=45)

ax.set_yticks(range(len(class_labels)))
ax.set_yticklabels(class_labels)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center"
        )

fig.colorbar(image, ax=ax)
fig.tight_layout()

fig.savefig(
    os.path.join(CONFUSION_FOLDER, "confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ================================================================
# 29. PERFORMANCE GRAPH
# ================================================================

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

metric_values = [
    accuracy * 100,
    precision * 100,
    recall * 100,
    f1 * 100
]

fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(metric_names, metric_values)

ax.set_title("AdaBoost - Employee Attrition Performance")
ax.set_xlabel("Evaluation Metric")
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 110)

for bar, value in zip(bars, metric_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

fig.tight_layout()

fig.savefig(
    os.path.join(CHARTS_FOLDER, "performance_graph.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ================================================================
# 30. ACTUAL VS PREDICTED CHART
# ================================================================

actual_counts = y_test.value_counts().reindex(
    class_labels, fill_value=0
)

predicted_counts = pd.Series(y_pred).value_counts().reindex(
    class_labels, fill_value=0
)

comparison_df = pd.DataFrame({
    "Attrition_Class": class_labels,
    "Actual": actual_counts.to_numpy(),
    "Predicted": predicted_counts.to_numpy()
})

comparison_df.to_csv(
    os.path.join(CHARTS_FOLDER, "actual_vs_predicted_counts.csv"),
    index=False
)

x = np.arange(len(class_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))

ax.bar(
    x - width / 2,
    comparison_df["Actual"],
    width,
    label="Actual"
)

ax.bar(
    x + width / 2,
    comparison_df["Predicted"],
    width,
    label="Predicted"
)

ax.set_xlabel("Employee Attrition Class")
ax.set_ylabel("Number of Employees")
ax.set_title("Actual vs Predicted Attrition - AdaBoost")
ax.set_xticks(x)
ax.set_xticklabels(class_labels)
ax.legend()

fig.tight_layout()

fig.savefig(
    os.path.join(CHARTS_FOLDER, "actual_vs_predicted.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ================================================================
# 31. ATTRITION CLASS DISTRIBUTION
# ================================================================

class_counts = y.value_counts().reindex(
    class_labels, fill_value=0
)

fig, ax = plt.subplots(figsize=(8, 6))

ax.bar(
    class_counts.index.astype(str),
    class_counts.values
)

ax.set_xlabel("Attrition Class")
ax.set_ylabel("Number of Employees")
ax.set_title("Employee Attrition Class Distribution")
ax.tick_params(axis="x", rotation=0)

fig.tight_layout()

fig.savefig(
    os.path.join(CHARTS_FOLDER, "class_distribution.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ================================================================
# 32. GET TRANSFORMED FEATURE NAMES
# ================================================================

feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

feature_names = [
    str(name) for name in feature_names
]


# ================================================================
# 33. FEATURE IMPORTANCE
# ================================================================

feature_importances = adaboost_classifier.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

feature_importance_df.to_csv(
    os.path.join(FEATURE_FOLDER, "feature_importance.csv"),
    index=False
)


# ================================================================
# 34. FEATURE IMPORTANCE GRAPH
# ================================================================

top_features = (
    feature_importance_df.head(15)
    .sort_values(by="Importance", ascending=True)
)

fig, ax = plt.subplots(figsize=(11, 7))

ax.barh(
    top_features["Feature"],
    top_features["Importance"]
)

ax.set_xlabel("Importance")
ax.set_ylabel("Employee Feature")
ax.set_title("Top 15 AdaBoost Feature Importances")

fig.tight_layout()

fig.savefig(
    os.path.join(FEATURE_FOLDER, "feature_importance.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ================================================================
# 35. FIRST ADABOOST BASE TREE VISUALIZATION
# ================================================================

print("\nDisplaying the first fitted AdaBoost decision tree...")

first_tree = adaboost_classifier.estimators_[0]

fig, ax = plt.subplots(figsize=(28, 16))

plot_tree(
    first_tree,
    feature_names=feature_names,
    class_names=[str(label) for label in class_labels],
    filled=True,
    rounded=True,
    precision=2,
    fontsize=8,
    ax=ax
)

ax.set_title("Skill Match - AdaBoost First Base Tree")

fig.tight_layout()

tree_image_path = os.path.join(
    TREE_FOLDER,
    "adaboost_base_tree_1.png"
)

fig.savefig(
    tree_image_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("Base tree saved:", tree_image_path)


# ================================================================
# 36. SAVE TEST PREDICTIONS
# ================================================================

test_predictions = X_test.copy()

test_predictions["Actual_Attrition"] = y_test.to_numpy()
test_predictions["Predicted_Attrition"] = y_pred

prediction_path = os.path.join(
    PREDICTIONS_FOLDER,
    "test_predictions.csv"
)

test_predictions.to_csv(
    prediction_path,
    index=False
)


# ================================================================
# 37. SAVE TRAINED PIPELINE
# ================================================================

model_path = os.path.join(
    MODEL_FOLDER,
    "adaboost_attrition_model.pkl"
)

with open(model_path, "wb") as file:
    pickle.dump(model, file)


# ================================================================
# 38. SAVE MODEL PARAMETERS
# ================================================================

parameters_df = pd.DataFrame({
    "Parameter": [
        "Algorithm",
        "Target Column",
        "Number of Estimators",
        "Learning Rate",
        "Base Tree Maximum Depth",
        "Train-Test Split",
        "Random State",
        "Numeric Preprocessing",
        "Categorical Encoding"
    ],
    "Value": [
        "AdaBoost Classifier",
        target_column,
        adaboost_classifier.n_estimators,
        adaboost_classifier.learning_rate,
        first_tree.get_depth(),
        "80% training / 20% testing",
        42,
        "Median Imputation",
        "One-Hot Encoding"
    ]
})

parameters_df.to_csv(
    os.path.join(METRICS_FOLDER, "adaboost_parameters.csv"),
    index=False
)


# ================================================================
# 39. SAVE ESTIMATOR WEIGHTS
# ================================================================

estimator_weights_df = pd.DataFrame({
    "Estimator_Number": np.arange(
        1,
        len(adaboost_classifier.estimator_weights_) + 1
    ),
    "Estimator_Weight": adaboost_classifier.estimator_weights_
})

estimator_weights_df.to_csv(
    os.path.join(METRICS_FOLDER, "estimator_weights.csv"),
    index=False
)


# ================================================================
# 40. FINAL SUMMARY
# ================================================================

print("\n" + "=" * 75)
print("       SKILL MATCH ADABOOST COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nOriginal RAW dataset was NOT modified.")
print("Dataset:", DATASET_PATH)

print("\nFINAL PERFORMANCE")
print("-" * 50)
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\nOUTPUT LOCATION")
print("-" * 50)
print(OUTPUT_FOLDER)

print("\nOUTPUT FOLDERS")
print("1. metrics")
print("2. predictions")
print("3. confusion_matrix")
print("4. charts")
print("5. adaboost_tree")
print("6. feature_importance")
print("7. model")

print("\n" + "=" * 75)
print("                    PROGRAM FINISHED")
print("=" * 75)
