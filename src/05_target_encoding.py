import pandas as pd
from category_encoders import TargetEncoder

# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = r"C:\Users\LEELA PRASAD\PycharmProjects\Skill_Match\dataset\reordered_raw.csv"

OUTPUT_FILE = r"C:\Users\LEELA PRASAD\PycharmProjects\Skill_Match\dataset\target_encoding.csv"

TARGET_COLUMN = "Attrition"


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ============================================================
# TARGET COLUMN
# ============================================================

print("\nTarget column:", TARGET_COLUMN)

# Convert Yes/No into 1/0
df[TARGET_COLUMN] = (
    df[TARGET_COLUMN]
    .astype(str)
    .str.strip()
    .map({
        "Yes": 1,
        "No": 0
    })
)

print("\nTarget values:")
print(df[TARGET_COLUMN].value_counts())


# ============================================================
# CATEGORICAL COLUMNS
# ============================================================

categorical_columns = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "Over18",
    "OverTime"
]

print("\n" + "=" * 60)
print("CATEGORICAL COLUMNS")
print("=" * 60)

for column in categorical_columns:
    print("-", column)


# ============================================================
# TARGET ENCODING
# ============================================================

encoder = TargetEncoder(
    cols=categorical_columns,
    handle_missing="value",
    handle_unknown="value"
)

# Convert categorical columns to string
df[categorical_columns] = df[categorical_columns].astype(str)

# Apply Target Encoding
df[categorical_columns] = encoder.fit_transform(
    df[categorical_columns],
    df[TARGET_COLUMN]
)


# ============================================================
# SAVE OUTPUT
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TARGET ENCODING COMPLETED")
print("=" * 60)

print("Output rows    :", df.shape[0])
print("Output columns :", df.shape[1])

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nFirst 5 rows:")
print(df.head())

print("\nFile saved successfully!")