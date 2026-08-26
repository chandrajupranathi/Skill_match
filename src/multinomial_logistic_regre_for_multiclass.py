import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder
)
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
print("MULTINOMIAL LOGISTIC REGRESSION")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# CHECK TARGET COLUMN
# ============================================================

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found."
    )

print("\nTarget column:", TARGET_COLUMN)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\nTarget values:")

print(
    df[TARGET_COLUMN].value_counts(
        dropna=False
    )
)


# ============================================================
# REMOVE MISSING TARGET VALUES
# ============================================================

df = df.dropna(
    subset=[TARGET_COLUMN]
).copy()


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[TARGET_COLUMN]
)

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
    column
    for column in columns_to_remove
    if column in X.columns
]

if existing_columns_to_remove:

    print("\nRemoving ID / constant columns:")

    print(
        existing_columns_to_remove
    )

    X = X.drop(
        columns=existing_columns_to_remove
    )


# ============================================================
# ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

number_of_classes = len(
    label_encoder.classes_
)


print(
    "\nNumber of target classes:",
    number_of_classes
)


print("\nTarget encoding:")

for original, encoded in zip(
    label_encoder.classes_,
    label_encoder.transform(
        label_encoder.classes_
    )
):

    print(
        f"{original} -> {encoded}"
    )


# ============================================================
# CHECK MULTICLASS CONDITION
# ============================================================

if number_of_classes < 3:

    print("\n" + "=" * 70)
    print("MULTICLASS MODEL NOT APPLICABLE")
    print("=" * 70)

    print(
        "\nThe target column contains only "
        f"{number_of_classes} classes."
    )

    print(
        "\nClasses found:"
    )

    print(
        list(label_encoder.classes_)
    )

    print(
        "\nMultinomial Logistic Regression is intended "
        "for a multiclass target with 3 or more classes."
    )

    print(
        "\nFor this dataset:"
    )

    print(
        "Use logistic_regression_binary_classify.py"
    )

    print(
        "\nBinary classification:"
    )

    print(
        "No  -> 0"
    )

    print(
        "Yes -> 1"
    )

    print("\n" + "=" * 70)

    raise SystemExit(0)


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
    include=[
        "object",
        "str",
        "category",
        "bool"
    ]
).columns.tolist()


print("\nNumerical columns:")

print(
    numerical_columns
)


print("\nCategorical columns:")

print(
    categorical_columns
)


# ============================================================
# NUMERICAL PIPELINE
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# CATEGORICAL PIPELINE
# ============================================================

categorical_pipeline = Pipeline(
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


# ============================================================
# COLUMN TRANSFORMER
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
# MULTINOMIAL LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    solver="lbfgs",
    multi_class="multinomial",
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


print(
    "\nTraining samples:",
    X_train.shape[0]
)

print(
    "Testing samples:",
    X_test.shape[0]
)


# ============================================================
# TRAIN MODEL
# ============================================================

print(
    "\nTraining Multinomial Logistic Regression..."
)

pipeline.fit(
    X_train,
    y_train
)

print(
    "Model training completed."
)


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

print(
    "\nClassification Report:"
)

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


print(
    "\nConfusion Matrix:"
)

print(cm)


# ============================================================
# DISPLAY CONFUSION MATRIX
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

disp.plot()

plt.title(
    "Multinomial Logistic Regression - Confusion Matrix"
)

plt.tight_layout()

plt.show()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("MULTINOMIAL LOGISTIC REGRESSION COMPLETED")
print("=" * 70)