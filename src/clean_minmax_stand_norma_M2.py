import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from pathlib import Path


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_FOLDER = PROJECT_DIR / "dataset"

INPUT_FILE = DATASET_FOLDER / "reordered_raw.csv"

OUTPUT_FILE = DATASET_FOLDER / "clean_minmax_stand_norma_M2.csv"


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)

print("Input file :", INPUT_FILE)
print("Rows       :", df.shape[0])
print("Columns    :", df.shape[1])

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\nMissing Values:")
print("------------------------")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())


# ============================================================
# 4. REMOVE DUPLICATE RECORDS
# ============================================================

df = df.drop_duplicates()

print("\nDataset Shape After Removing Duplicates:")
print(df.shape)


# ============================================================
# 5. IDENTIFY NUMERIC COLUMNS
# ============================================================

numerical_columns = df.select_dtypes(
    include="number"
).columns.tolist()

print("\nNumeric Columns:")
print(numerical_columns)


# ============================================================
# 6. IDENTIFY CATEGORICAL COLUMNS
# ============================================================

categorical_columns = [
    column
    for column in df.columns
    if column not in numerical_columns
]

print("\nCategorical Columns:")
print(categorical_columns)


# ============================================================
# 7. HANDLE MISSING NUMERIC VALUES
#    Missing values -> MEAN
# ============================================================

for column in numerical_columns:

    df[column] = df[column].fillna(
        df[column].mean()
    )


# ============================================================
# 8. HANDLE MISSING CATEGORICAL VALUES
#    Missing values -> MODE
# ============================================================

for column in categorical_columns:

    df[column] = df[column].fillna(
        df[column].mode()[0]
    )


# ============================================================
# 9. REMOVE LEADING/TRAILING SPACES
# ============================================================

for column in categorical_columns:

    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
    )


# ============================================================
# 10. REMOVE USELESS COLUMNS
# ============================================================

columns_to_remove = [
    "EmployeeCount",
    "EmployeeNumber",
    "StandardHours",
    "Over18"
]

# Remove only columns that exist
columns_to_remove = [
    column
    for column in columns_to_remove
    if column in df.columns
]

df = df.drop(
    columns=columns_to_remove
)

print("\nRemoved Useless Columns:")
print(columns_to_remove)


# ============================================================
# 11. SELECT NUMERIC COLUMNS AGAIN
# ============================================================

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()

print("\nNumeric Columns Used for Scaling:")
print(numeric_columns)


# ============================================================
# 12. STANDARDIZATION
#    Mean = 0
#    Standard Deviation = 1
# ============================================================

standard_scaler = StandardScaler()

standardized = standard_scaler.fit_transform(
    df[numeric_columns]
)

standardized_df = pd.DataFrame(
    standardized,
    columns=[
        column + "_Standardized"
        for column in numeric_columns
    ],
    index=df.index
)


# ============================================================
# 13. MIN-MAX SCALING
#    Values between 0 and 1
# ============================================================

minmax_scaler = MinMaxScaler()

scaled = minmax_scaler.fit_transform(
    df[numeric_columns]
)

scaled_df = pd.DataFrame(
    scaled,
    columns=[
        column + "_Scaled"
        for column in numeric_columns
    ],
    index=df.index
)


# ============================================================
# 14. NORMALIZATION
#    L2 Normalization
# ============================================================

normalizer = Normalizer(
    norm="l2"
)

normalized = normalizer.fit_transform(
    df[numeric_columns]
)

normalized_df = pd.DataFrame(
    normalized,
    columns=[
        column + "_Normalized"
        for column in numeric_columns
    ],
    index=df.index
)


# ============================================================
# 15. COMBINE ALL DATA
# ============================================================

df = pd.concat(
    [
        df,
        standardized_df,
        scaled_df,
        normalized_df
    ],
    axis=1
)


# ============================================================
# 16. CHECK FINAL DATASET
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSED DATASET")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing Values After Preprocessing:")
print(df.isnull().sum().sum())

print("\nDuplicate Records After Preprocessing:")
print(df.duplicated().sum())


# ============================================================
# 17. SAVE DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 18. FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Input file:")
print(INPUT_FILE)

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nFinal Shape:")
print(df.shape)

print("\nFile saved successfully!")