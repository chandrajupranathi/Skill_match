import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ============================================================
# FILE PATH
# ============================================================

INPUT_FILE = r"C:\Users\LEELA PRASAD\PycharmProjects\Skill_Match\dataset\cleaned_mean_model_imputed.csv"

OUTPUT_FILE = r"C:\Users\LEELA PRASAD\PycharmProjects\Skill_Match\dataset\label_encoded.csv"


# ============================================================
# READ DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


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
    "OverTime",
    "Attrition"
]


print("\nCategorical columns:")

for column in categorical_columns:
    print("-", column)


# ============================================================
# LABEL ENCODING
# ============================================================

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(
        df[column].astype(str)
    )


# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 60)
print("LABEL ENCODING COMPLETED")
print("=" * 60)

print("Original shape : 1470 rows × 35 columns")
print("Output shape   :", df.shape)

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nFirst 5 rows:")
print(df.head())

print("\nFile saved successfully!")