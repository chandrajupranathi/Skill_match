# ==============================================================
# SKILL MATCH (SM)
# DATASET LOADING AND MISSING VALUE IDENTIFICATION - M1
#
# Original dataset is NOT modified.
# All outputs are saved inside the outputs folder.
# ==============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# 1. Define project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_DIR / "dataset" / "reordered_raw.csv"

OUTPUT_FOLDER = (
    PROJECT_DIR / "output" / "Dataset_Load_Missing_Values_M1_Outputs"
)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
# 2. Load the dataset
df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("SKILL MATCH - DATASET LOADING AND MISSING VALUE ANALYSIS")
print("=" * 60)

print("\nDataset loaded successfully!")
print("Dataset path:", DATASET_PATH)
print("Dataset shape:", df.shape)


# 3. View the first 5 rows
print("\n--- First 5 Rows ---")
print(df.head())


# 4. Print the first 6 columns
print("\n--- Print 6 Columns ---")
subset = df.iloc[:, 0:6]
print(subset)


# 5. Identify missing values per column
missing_counts = df.isnull().sum()

print("\n--- Missing Values Per Column ---")
print(missing_counts)
print("-" * 40)

print("Total missing values:", df.isnull().sum().sum())


# 6. Detect duplicate rows
duplicate_rows = df[df.duplicated()]

print("\n--- Duplicate Row Analysis ---")
print(f"Total duplicate rows detected: {len(duplicate_rows)}")
print(duplicate_rows)
print("-" * 40)


# 7. Produce a missingness heatmap
plt.figure(figsize=(10, 6))

sns.heatmap(
    df.isnull(),
    cbar=False,
    yticklabels=False,
    cmap="viridis"
)

plt.title("Skill Match - Missing Values Heatmap")
plt.xlabel("Dataset Columns")
plt.ylabel("Dataset Records")
plt.tight_layout()

# Save the heatmap in the output folder
heatmap_path = OUTPUT_FOLDER / "missing_values_heatmap.png"
plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")

# Close the figure without triggering the PyCharm
# non-interactive FigureCanvasAgg warning
plt.close()


# 8. Save the missing-value report
missing_report = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": df.isnull().sum().values,
    "Missing_Percentage": (
        df.isnull().sum().values / len(df) * 100
    )
})

report_path = OUTPUT_FOLDER / "missing_values_report.csv"
missing_report.to_csv(report_path, index=False)


# 9. Final summary
print("\n--- OUTPUT FILES ---")
print("Missing values heatmap:", heatmap_path)
print("Missing values report:", report_path)

print("\nDATASET LOADING AND MISSING VALUE ANALYSIS COMPLETED!")
print("Output folder:", OUTPUT_FOLDER)
print("=" * 60)
