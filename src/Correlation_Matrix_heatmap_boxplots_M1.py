import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_FOLDER = PROJECT_DIR / "dataset"

OUTPUT_DIR = PROJECT_DIR / "output" / "Correlation_heatmap"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD EMPLOYEE ATTRITION DATASET
# ============================================================

INPUT_FILE = DATASET_FOLDER / "reordered_raw.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)

print("Input file :", INPUT_FILE)
print("Shape      :", df.shape)


# ============================================================
# 3. SELECT NUMERICAL COLUMNS
# ============================================================

numerical_cols = df.select_dtypes(
    include=np.number
).columns.tolist()


# ============================================================
# 4. REMOVE ID / CONSTANT COLUMNS
# ============================================================

columns_to_remove = [
    "EmployeeNumber",
    "EmployeeCount",
    "StandardHours"
]

numerical_cols = [
    col
    for col in numerical_cols
    if col not in columns_to_remove
]


print("\nNumerical columns used:")
print(numerical_cols)


# ============================================================
# 5. CORRELATION MATRIX
# ============================================================

corr_matrix = df[numerical_cols].corr()


print("\n" + "=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)

print(corr_matrix)


# ============================================================
# 6. CREATE CORRELATION HEATMAP
# ============================================================

plt.figure(
    figsize=(14, 10)
)

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5
)

plt.title(
    "Correlation Heatmap of Numerical Features",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()


# ============================================================
# 7. SAVE CORRELATION HEATMAP
# ============================================================

heatmap_path = OUTPUT_DIR / "correlation_heatmap.png"

plt.savefig(
    heatmap_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nCorrelation heatmap saved to:")
print(heatmap_path)


# ============================================================
# 8. BOXPLOTS: NUMERICAL FEATURES VS ATTRITION
# ============================================================

target_col = "Attrition"


if target_col in df.columns:

    print("\n" + "=" * 60)
    print("CREATING BOXPLOTS")
    print("=" * 60)

    for col in numerical_cols:

        plt.figure(
            figsize=(6, 5)
        )

        sns.boxplot(
            x=target_col,
            y=col,
            data=df,
            hue=target_col,
            palette="Set2",
            legend=False
        )

        plt.title(
            f"{col} vs {target_col}",
            fontsize=12,
            fontweight="bold"
        )

        plt.xlabel(target_col)
        plt.ylabel(col)

        plt.tight_layout()


        # ====================================================
        # SAVE BOXPLOT
        # ====================================================

        boxplot_filename = (
            f"boxplot_{col}_vs_{target_col}.png"
        )

        boxplot_path = (
            OUTPUT_DIR / boxplot_filename
        )

        plt.savefig(
            boxplot_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {boxplot_filename}"
        )


else:

    print(
        f"\nTarget column '{target_col}' "
        "not found in dataset."
    )

    print("Skipping boxplots.")


# ============================================================
# 9. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("ALL EDA TASKS COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nOutput folder:")
print(OUTPUT_DIR)