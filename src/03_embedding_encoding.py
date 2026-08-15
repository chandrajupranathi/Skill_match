import pandas as pd


# ============================================================
# FILE PATH
# ============================================================

INPUT_FILE = r"C:\Users\LEELA PRASAD\PycharmProjects\Skill_Match\dataset\cleaned_mean_model_imputed.csv"

OUTPUT_FILE = r"C:\Users\LEELA PRASAD\PycharmProjects\Skill_Match\dataset\embedding_encoding.csv"


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


# ============================================================
# EMBEDDING / CATEGORY ID ENCODING
# ============================================================

for column in categorical_columns:

    # Get unique categories
    categories = df[column].astype(str).unique()

    # Give each category a number
    mapping = {
        category: index
        for index, category in enumerate(categories)
    }

    # Replace categories with their numbers
    df[column] = (
        df[column]
        .astype(str)
        .map(mapping)
    )


# ============================================================
# SAVE ENCODED DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 60)
print("EMBEDDING ENCODING COMPLETED")
print("=" * 60)

print("Output shape:", df.shape)

print("\nOutput file:")
print(OUTPUT_FILE)

print("\nFirst 5 rows:")
print(df.head())

print("\nFile saved successfully!")