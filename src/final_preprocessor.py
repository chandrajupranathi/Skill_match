# ============================================================
# FINAL PREPROCESSOR
# Skill_Match Dataset
# ============================================================

import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. PROJECT PATH
# ============================================================

DATASET_FOLDER = Path(
    r"C:\Users\LEELA PRASAD\PycharmProjects\Skill_Match\dataset"
)

OUTPUT_FILE = DATASET_FOLDER / "final_preprocessed.csv"

CLEANED_FILE = DATASET_FOLDER / "cleaned_mean_model_imputed.csv"


# ============================================================
# 2. CHECK INPUT FILE
# ============================================================

if not CLEANED_FILE.exists():

    raise FileNotFoundError(
        f"Cleaned dataset not found:\n{CLEANED_FILE}\n\n"
        "Run clean_del_mean_model_missing_imputer_M2.py first."
    )


INPUT_FILE = CLEANED_FILE


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("=" * 60)
print("FINAL PREPROCESSOR")
print("=" * 60)

print("Input file:")
print(INPUT_FILE)


df = pd.read_csv(INPUT_FILE)


print("\nDataset loaded successfully!")

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ============================================================
# 4. TARGET VARIABLE
# ============================================================

TARGET_COLUMN = "Attrition"


if TARGET_COLUMN not in df.columns:

    raise ValueError(
        f"Target column '{TARGET_COLUMN}' was not found."
    )


# ============================================================
# 5. CHECK TARGET
# ============================================================

print("\nTarget values:")

print(
    df[TARGET_COLUMN].value_counts(
        dropna=False
    )
)


# ============================================================
# 6. CONVERT TARGET TO NUMERIC
# ============================================================

# No  -> 0
# Yes -> 1

df[TARGET_COLUMN] = df[TARGET_COLUMN].map({
    "No": 0,
    "Yes": 1
})


# ============================================================
# 7. CHECK TARGET VALUES
# ============================================================

if df[TARGET_COLUMN].isna().any():

    raise ValueError(
        "Target column contains missing or invalid values."
    )


valid_targets = set(
    df[TARGET_COLUMN].unique()
)


if not valid_targets.issubset({0, 1}):

    raise ValueError(
        f"Unexpected target values found: {valid_targets}"
    )


print("\nTarget verified successfully.")

print(
    df[TARGET_COLUMN].value_counts()
)


# ============================================================
# 8. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[TARGET_COLUMN]
)

y = df[TARGET_COLUMN]


# ============================================================
# 9. FIND NUMERIC FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()


# ============================================================
# 10. FIND CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    column
    for column in X.columns
    if column not in numeric_features
]


print("\n" + "=" * 60)
print("FEATURE INFORMATION")
print("=" * 60)


print("\nNumeric features:")

for column in numeric_features:

    print("-", column)


print("\nCategorical features:")

for column in categorical_features:

    print("-", column)


# ============================================================
# 11. NUMERIC PIPELINE
# ============================================================

numeric_pipeline = Pipeline(
    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="mean"
            )
        ),

        (
            "scaler",

            StandardScaler()
        )
    ]
)


# ============================================================
# 12. CATEGORICAL PIPELINE
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
            "onehot",

            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ============================================================
# 13. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",

            numeric_pipeline,

            numeric_features
        ),

        (
            "categorical",

            categorical_pipeline,

            categorical_features
        )
    ]
)


# ============================================================
# 14. FIT AND TRANSFORM
# ============================================================

X_processed = preprocessor.fit_transform(X)


print("\nPreprocessing completed successfully.")


# ============================================================
# 15. FEATURE NAMES
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)


# ============================================================
# 16. CREATE FINAL DATAFRAME
# ============================================================

X_processed_df = pd.DataFrame(

    X_processed,

    columns=feature_names
)


# ============================================================
# 17. ADD TARGET
# ============================================================

X_processed_df[TARGET_COLUMN] = y.values


# ============================================================
# 18. CHECK MISSING VALUES
# ============================================================

missing_values = (

    X_processed_df
    .isnull()
    .sum()
    .sum()
)


print("\nMissing values in final dataset:")

print(missing_values)


# ============================================================
# 19. SAVE FINAL DATASET
# ============================================================

X_processed_df.to_csv(

    OUTPUT_FILE,

    index=False
)


# ============================================================
# 20. FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("FINAL PREPROCESSING COMPLETED")
print("=" * 60)


print("\nOriginal shape:")

print(df.shape)


print("\nFinal shape:")

print(X_processed_df.shape)


print("\nTarget distribution:")

print(
    X_processed_df[TARGET_COLUMN]
    .value_counts()
)


print("\nOutput file:")

print(OUTPUT_FILE)


print("\nFirst 5 rows:")

print(
    X_processed_df.head()
)


print("\nFinal preprocessed CSV saved successfully!")