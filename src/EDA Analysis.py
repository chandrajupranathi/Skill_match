import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. PATH SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_DIR / "dataset" / "reordered_raw.csv"

OUTPUT_FOLDER = PROJECT_DIR / "output" / "EDA_Analysis_outputs2"

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. GRAPH SETTINGS
# ============================================================

sns.set(style="whitegrid")

plt.rcParams["figure.figsize"] = (8, 5)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("=" * 60)
print("EDA ANALYSIS")
print("=" * 60)

print("\nLoading Dataset...")

df = pd.read_csv(DATASET_PATH)


# ============================================================
# 4. BASIC DATASET INFORMATION
# ============================================================

print("\nFirst Five Records")
print("-" * 60)

print(df.head())


print("\nDataset Shape:")
print(df.shape)


print("\nData Types")
print("-" * 60)

print(df.dtypes)


print("\nDataset Information")
print("-" * 60)

df.info()


print("\nMissing Values")
print("-" * 60)

print(df.isnull().sum())


print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 5. NUMERICAL COLUMNS
# ============================================================

numeric_cols = df.select_dtypes(
    include=["number"]
).columns.tolist()


print("\nNumerical Columns")
print("-" * 60)

print(numeric_cols)


# ============================================================
# 6. CATEGORICAL COLUMNS
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


print("\nCategorical Columns")
print("-" * 60)

print(categorical_columns)


# ============================================================
# 7. TARGET COLUMN
# ============================================================

target = None

possible_targets = [
    "Attrition",
    "PlacementStatus",
    "Placement",
    "Status",
    "Placed"
]


for col in possible_targets:

    if col in df.columns:

        target = col

        break


print("\nTarget Column:")

if target:

    print(target)

else:

    print("No target column found")


# ============================================================
# 8. STATISTICAL SUMMARY
# ============================================================

print("\nStatistical Summary")
print("-" * 60)

summary = df.describe(include="all")

print(summary)


summary.to_csv(
    OUTPUT_FOLDER / "Statistical_Summary.csv"
)


# ============================================================
# 9. CORRELATION MATRIX
# ============================================================

print("\nCreating Correlation Matrix...")


numeric_df = df.select_dtypes(
    include=np.number
)


corr = numeric_df.corr()


# Save correlation matrix
corr.to_csv(
    OUTPUT_FOLDER / "Correlation_Matrix.csv"
)


# Create heatmap
plt.figure(figsize=(14, 10))


sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)


plt.title("Correlation Matrix")

plt.tight_layout()


plt.savefig(
    OUTPUT_FOLDER / "Correlation_Heatmap.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 10. HISTOGRAMS
# ============================================================

print("\nCreating Histograms...")


for col in numeric_cols:

    plt.figure(figsize=(8, 5))


    sns.histplot(
        df[col],
        bins=20,
        kde=True
    )


    plt.title(f"Histogram - {col}")

    plt.xlabel(col)

    plt.ylabel("Frequency")


    plt.tight_layout()


    plt.savefig(
        OUTPUT_FOLDER / f"Histogram_{col}.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# 11. BOXPLOTS
# ============================================================

print("\nCreating Boxplots...")


for col in numeric_cols:

    plt.figure(figsize=(6, 4))


    sns.boxplot(
        y=df[col]
    )


    plt.title(f"Box Plot - {col}")

    plt.ylabel(col)


    plt.tight_layout()


    plt.savefig(
        OUTPUT_FOLDER / f"Boxplot_{col}.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# 12. OUTLIER DETECTION
# ============================================================

print("\nCreating Outlier Plots...")


for col in numeric_cols:

    plt.figure(figsize=(6, 4))


    sns.boxplot(
        y=df[col]
    )


    plt.title(f"Outlier Detection - {col}")


    plt.tight_layout()


    plt.savefig(
        OUTPUT_FOLDER / f"Outlier_{col}.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# 13. CATEGORICAL COUNT PLOTS
# ============================================================

print("\nCreating Categorical Count Plots...")


for col in categorical_columns:

    plt.figure(figsize=(8, 5))


    sns.countplot(
        data=df,
        x=col
    )


    plt.xticks(rotation=45)

    plt.title(f"Count Plot - {col}")

    plt.xlabel(col)

    plt.ylabel("Count")


    plt.tight_layout()


    plt.savefig(
        OUTPUT_FOLDER / f"Countplot_{col}.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# 14. PAIRPLOT
# ============================================================

print("\nCreating Pairplot...")


if len(numeric_df.columns) > 1:

    pair = sns.pairplot(
        numeric_df
    )


    pair.savefig(
        OUTPUT_FOLDER / "Pairplot.png",
        dpi=300
    )


    plt.close("all")


# ============================================================
# 15. MISSING VALUES HEATMAP
# ============================================================

print("\nCreating Missing Values Heatmap...")


plt.figure(figsize=(12, 6))


sns.heatmap(
    df.isnull(),
    cbar=False,
    cmap="viridis"
)


plt.title("Missing Values Heatmap")


plt.tight_layout()


plt.savefig(
    OUTPUT_FOLDER / "MissingValues_Heatmap.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 16. SCATTER PLOT
# ============================================================

print("\nCreating Scatter Plot...")


if len(numeric_df.columns) >= 2:

    x_col = numeric_df.columns[0]

    y_col = numeric_df.columns[1]


    plt.figure(figsize=(8, 6))


    if target is not None:

        sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            hue=target,
            s=80
        )

    else:

        sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            s=80
        )


    plt.title(
        f"Scatter Plot: {x_col} vs {y_col}"
    )


    plt.xlabel(x_col)

    plt.ylabel(y_col)

    plt.grid(True)

    plt.tight_layout()


    plt.savefig(
        OUTPUT_FOLDER / "Scatter_Plot.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# 17. TARGET DISTRIBUTION
# ============================================================

if target is not None:

    print("\nCreating Target Distribution...")


    plt.figure(figsize=(7, 5))


    sns.countplot(
        data=df,
        x=target
    )


    plt.title(
        f"{target} Distribution"
    )


    plt.xlabel(target)

    plt.ylabel("Number of Employees")


    plt.tight_layout()


    plt.savefig(
        OUTPUT_FOLDER / "Target_Distribution.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# 18. FEATURE VS TARGET BOXPLOTS
# ============================================================

if target is not None:

    print("\nCreating Feature vs Target Boxplots...")


    for col in numeric_cols:

        plt.figure(figsize=(8, 5))


        sns.boxplot(
            x=df[target],
            y=df[col]
        )


        plt.title(
            f"{col} vs {target}"
        )


        plt.xlabel(target)

        plt.ylabel(col)


        plt.tight_layout()


        plt.savefig(
            OUTPUT_FOLDER / f"{col}_vs_{target}.png",
            dpi=300,
            bbox_inches="tight"
        )


        plt.close()


# ============================================================
# 19. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 60)

print("EDA COMPLETED SUCCESSFULLY")

print("=" * 60)


print("\nAll figures are saved in:")

print(OUTPUT_FOLDER)