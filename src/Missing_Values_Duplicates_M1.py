import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. Load the Dataset
# ============================================================

file_path = r"C:\Users\LEELA PRASAD\PycharmProjects\Skill_Match\dataset\reordered_raw.csv"

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)


# ============================================================
# 2. Retrieve Data in Different Ways
# ============================================================

# View the first 5 rows

print("\n--- First 5 Rows ---")
print(df.head())


# Print first 6 columns

print("\n--- First 6 Columns ---")

subset = df.iloc[:, 0:6]

print(subset)


# ============================================================
# 3. Identify Missing Values Per Column
# ============================================================

missing_counts = df.isnull().sum()

print("\n----- Missing Values Per Column -----")
print(missing_counts)


# ============================================================
# 4. Total Missing Values
# ============================================================

total_missing = df.isnull().sum().sum()

print("\n-----------------------------------")
print("Total Missing Values:", total_missing)
print("-----------------------------------")


# ============================================================
# 5. Detect Duplicate Rows
# ============================================================

duplicate_rows = df[df.duplicated()]

print(
    f"\nTotal Duplicate Rows Detected: "
    f"{len(duplicate_rows)}"
)

print("\nDuplicate Rows:")
print(duplicate_rows)

print("-" * 40)


# ============================================================
# 6. Produce a Missingness Heatmap
# ============================================================

plt.figure(figsize=(10, 6))

sns.heatmap(
    df.isnull(),
    cbar=False,
    yticklabels=False,
    cmap="viridis"
)

plt.title("Missing Values Heatmap")

plt.tight_layout()

plt.show()


# ============================================================
# 7. Completion Message
# ============================================================

print("\nMissing Value and Duplicate Analysis Completed Successfully!")