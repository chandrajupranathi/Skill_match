
# ============================================================
# SKILL MATCH - RANDOM FOREST CLASSIFICATION
# Employee Attrition Prediction
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. FILE PATHS
# ============================================================

DATASET_PATH = "dataset/reordered_raw.csv"

OUTPUT_DIR = "outputs/Random_Forest"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("SKILL MATCH - RANDOM FOREST")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully!")
print("Dataset Shape:", df.shape)


# ============================================================
# 3. DISPLAY BASIC INFORMATION
# ============================================================

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum().sum())


# ============================================================
# 4. TARGET COLUMN
# ============================================================

TARGET_COLUMN = "Attrition"

print("\nTarget Column:", TARGET_COLUMN)

print("\nTarget Distribution:")
print(df[TARGET_COLUMN].value_counts())


# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]


# ============================================================
# 6. CONVERT TARGET INTO 0 AND 1
# ============================================================

# Yes = 1
# No  = 0

y = y.map({
    "No": 0,
    "Yes": 1
})

print("\nEncoded Target Distribution:")
print(y.value_counts())


# ============================================================
# 7. IDENTIFY CATEGORICAL AND NUMERICAL COLUMNS
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical Columns:")
print(categorical_columns)

print("\nNumerical Columns:")
print(numerical_columns)


# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape :", X_test.shape)


# ============================================================
# 9. ONE-HOT ENCODING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# ============================================================
# 10. RANDOM FOREST MODEL
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=200,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# ============================================================
# 11. CREATE PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", random_forest)
    ]
)


# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Random Forest training completed!")


# ============================================================
# 13. PREDICTION
# ============================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ============================================================
# 14. EVALUATION METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ============================================================
# 15. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST RESULTS")
print("=" * 70)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Attrition", "Attrition"],
        zero_division=0
    )
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 18. SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(figsize=(7, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Attrition", "Attrition"]
)

disp.plot()

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

confusion_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(confusion_path, dpi=300)
plt.close()

print("\nConfusion matrix saved:")
print(confusion_path)


# ============================================================
# 19. FEATURE IMPORTANCE
# ============================================================

# Get the trained Random Forest
rf_model = model.named_steps["classifier"]

# Get feature names after One-Hot Encoding
encoded_features = model.named_steps[
    "preprocessor"
].get_feature_names_out()

# Get feature importance
importance = rf_model.feature_importances_


# ============================================================
# 20. CREATE FEATURE IMPORTANCE DATAFRAME
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": encoded_features,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


# ============================================================
# 21. DISPLAY TOP 20 FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 IMPORTANT FEATURES")
print("=" * 70)

print(
    feature_importance.head(20).to_string(index=False)
)


# ============================================================
# 22. SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance_path = os.path.join(
    OUTPUT_DIR,
    "feature_importance.csv"
)

feature_importance.to_csv(
    feature_importance_path,
    index=False
)

print("\nFeature importance saved:")
print(feature_importance_path)


# ============================================================
# 23. FEATURE IMPORTANCE GRAPH
# ============================================================

top_features = feature_importance.head(20)

plt.figure(figsize=(10, 8))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest - Top 20 Feature Importance")

plt.tight_layout()

feature_plot_path = os.path.join(
    OUTPUT_DIR,
    "feature_importance.png"
)

plt.savefig(
    feature_plot_path,
    dpi=300
)

plt.close()

print("\nFeature importance graph saved:")
print(feature_plot_path)


# ============================================================
# 24. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

metrics_path = os.path.join(
    OUTPUT_DIR,
    "random_forest_metrics.csv"
)

metrics.to_csv(
    metrics_path,
    index=False
)

print("\nMetrics saved:")
print(metrics_path)


# ============================================================
# 25. SAVE PREDICTIONS
# ============================================================

prediction_results = X_test.copy()

prediction_results["Actual_Attrition"] = y_test.values

prediction_results["Predicted_Attrition"] = y_pred

prediction_results["Attrition_Probability"] = y_probability

prediction_path = os.path.join(
    OUTPUT_DIR,
    "random_forest_predictions.csv"
)

prediction_results.to_csv(
    prediction_path,
    index=False
)

print("\nPredictions saved:")
print(prediction_path)


# ============================================================
# 26. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nGenerated files:")
print("1. confusion_matrix.png")
print("2. feature_importance.csv")
print("3. feature_importance.png")
print("4. random_forest_metrics.csv")
print("5. random_forest_predictions.csv")

