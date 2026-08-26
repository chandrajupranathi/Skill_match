import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# FILE PATH
# ============================================================

INPUT_FILE = "../dataset/reordered_raw.csv"

TARGET_COLUMN = "Attrition"


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("BINARY LOGISTIC REGRESSION")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# CHECK TARGET COLUMN
# ============================================================

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found in dataset."
    )

print("\nTarget column:", TARGET_COLUMN)

print("\nTarget values:")
print(df[TARGET_COLUMN].value_counts(dropna=False))


# ============================================================
# REMOVE ROWS WITH MISSING TARGET
# ============================================================

df = df.dropna(subset=[TARGET_COLUMN]).copy()


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]


# ============================================================
# REMOVE ID / CONSTANT COLUMNS
# ============================================================

columns_to_remove = [
    "EmployeeNumber",
    "EmployeeCount",
    "StandardHours"
]

existing_columns_to_remove = [
    column for column in columns_to_remove
    if column in X.columns
]

if existing_columns_to_remove:

    print("\nRemoving ID / constant columns:")
    print(existing_columns_to_remove)

    X = X.drop(
        columns=existing_columns_to_remove
    )


# ============================================================
# ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nTarget encoding:")

for original, encoded in zip(
    label_encoder.classes_,
    label_encoder.transform(label_encoder.classes_)
):
    print(f"{original} -> {encoded}")


# ============================================================
# CHECK NUMBER OF CLASSES
# ============================================================

number_of_classes = len(label_encoder.classes_)

if number_of_classes != 2:

    raise ValueError(
        f"Binary Logistic Regression requires exactly 2 classes. "
        f"Found {number_of_classes} classes."
    )


# ============================================================
# IDENTIFY NUMERICAL COLUMNS
# ============================================================

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()


# ============================================================
# IDENTIFY CATEGORICAL COLUMNS
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object", "str", "category", "bool"]
).columns.tolist()


print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)


# ============================================================
# NUMERICAL PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
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


# ============================================================
# COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numeric_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ],
    remainder="drop"
)


# ============================================================
# LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)


# ============================================================
# COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            model
        )
    ]
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Binary Logistic Regression...")

pipeline.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ============================================================
# PREDICTION
# ============================================================

y_pred = pipeline.predict(
    X_test
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"Accuracy Percentage: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# CONFUSION MATRIX DETAILS
# ============================================================

tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix Details:")
print("True Negative  (TN):", tn)
print("False Positive (FP):", fp)
print("False Negative (FN):", fn)
print("True Positive  (TP):", tp)


# ============================================================
# DISPLAY CONFUSION MATRIX
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

disp.plot()

plt.title(
    "Binary Logistic Regression - Confusion Matrix"
)

plt.tight_layout()

plt.show()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("BINARY LOGISTIC REGRESSION COMPLETED")
print("=" * 70)