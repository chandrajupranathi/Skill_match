import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================================
# 1. Load Dataset
# ==========================================================

print("1. Load the Dataset")

# Get project folder automatically
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Dataset path
file_path = PROJECT_DIR / "dataset" / "reordered_raw.csv"

# Single output folder for all graphs
OUTPUT_FOLDER = (
    PROJECT_DIR / "output" / "Data_Understanding_Outputs"
)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

try:
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Display settings
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Display complete dataset
    print("-" * 80)
    print("1. Dataset Contents")
    print("-" * 80)
    print(df)

    # Dataset Shape
    print("-" * 80)
    print("2. Number of Rows and Columns:", df.shape)

    # Column Names
    print("-" * 80)
    print("3. Column Names")
    print(df.columns.tolist())

    # First 10 Rows
    print("-" * 80)
    print("4. First 10 Records")
    print("-" * 80)
    print(df.head(10))

    # Last 10 Rows
    print("-" * 80)
    print("Last 10 Records")
    print("-" * 80)
    print(df.tail(10))

    # ==========================================================
    # 2. Understand Dataset
    # ==========================================================

    print("\n" + "=" * 80)
    print("2. Understand the Dataset")
    print("=" * 80)

    # Data Types
    print("\n5. Data Types")
    print(df.dtypes)

    # Column Names with Data Types
    print("\n6. Column Names with Data Types")
    print("-" * 40)

    for column in df.columns:
        print(f"{column:<25} {df[column].dtype}")

    # Dataset Information
    print("\n" + "=" * 80)
    print("7. Dataset Information")
    print("=" * 80)

    df.info()

    # ==========================================================
    # Numeric Columns
    # ==========================================================

    print("\n8. Numeric Columns")

    numeric_df = df.select_dtypes(include=["number"])
    print(numeric_df)

    print("\n9. Missing Values in Numeric Columns")
    print(numeric_df.isnull().sum())

    print(
        "\nTotal Missing Numeric Values:",
        numeric_df.isnull().sum().sum()
    )

    # Float Columns
    float_columns = df.select_dtypes(include=["float"]).columns

    print("\n10. Float Column Names")
    print("-" * 40)

    for column in float_columns:
        print(column)

    print("\n11. Missing Values in Float Columns")
    print(df[float_columns].isnull().sum())

    print(
        "\nTotal Missing Float Values:",
        df[float_columns].isnull().sum().sum()
    )

    # ==========================================================
    # Categorical Columns
    # ==========================================================

    categorical_df = df.select_dtypes(
        include=["object", "category", "string"]
    )

    print("\n12. Categorical Columns")
    print("-" * 40)
    print(categorical_df)

    print("\nMissing Values in Categorical Columns")
    print(categorical_df.isnull().sum())

    print(
        "\nTotal Missing Categorical Values:",
        categorical_df.isnull().sum().sum()
    )

    # ==========================================================
    # Missing Values
    # ==========================================================

    print("\n13. Missing Values in Each Column")
    print(df.isnull().sum())

    total_missing = df.isnull().sum().sum()

    print("\n14. Total Missing Values:", total_missing)

    # ==========================================================
    # Duplicate Records
    # ==========================================================

    duplicate_count = df.duplicated().sum()

    print("\n15. Duplicate Records:", duplicate_count)

    # ==========================================================
    # Statistical Summary
    # ==========================================================

    print("\n16. Statistical Summary")
    print(df.describe())

    # ==========================================================
    # Histogram (Age)
    # ==========================================================

    if "Age" in df.columns:
        print("\n17. Histogram of Age")

        plt.figure(figsize=(8, 5))

        plt.hist(
            df["Age"].dropna(),
            bins=10,
            edgecolor="black"
        )

        plt.title("Histogram of Employee Age")
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            OUTPUT_FOLDER / "age_histogram.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()
    else:
        print("\nAge column not found in dataset.")

    # ==========================================================
    # Correlation Heatmap
    # ==========================================================

    if not numeric_df.empty:
        correlation_df = numeric_df.corr()

        figure_size = max(
            10,
            len(correlation_df.columns) * 0.7
        )

        plt.figure(figsize=(figure_size, figure_size))

        sns.heatmap(
            correlation_df,
            annot=len(correlation_df.columns) <= 15,
            cmap="coolwarm",
            fmt=".2f"
        )

        plt.title("Skill Match - Correlation Heatmap")
        plt.tight_layout()

        plt.savefig(
            OUTPUT_FOLDER / "correlation_heatmap.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()

    # ==========================================================
    # Attrition Distribution
    # ==========================================================

    if "Attrition" in df.columns:
        print("\n18. Attrition Target Distribution")
        print(df["Attrition"].value_counts(dropna=False))

        plt.figure(figsize=(7, 5))

        sns.countplot(
            data=df,
            x="Attrition"
        )

        plt.title("Employee Attrition Distribution")
        plt.xlabel("Attrition")
        plt.ylabel("Number of Employees")
        plt.tight_layout()

        plt.savefig(
            OUTPUT_FOLDER / "attrition_distribution.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()
    else:
        print("\nAttrition column not found in dataset.")

    # ==========================================================
    # Histogram (Monthly Income)
    # ==========================================================

    if "MonthlyIncome" in df.columns:
        print("\n19. Histogram of Monthly Income")

        plt.figure(figsize=(8, 5))

        plt.hist(
            df["MonthlyIncome"].dropna(),
            bins=10,
            edgecolor="black"
        )

        plt.title("Histogram of Monthly Income")
        plt.xlabel("Monthly Income")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            OUTPUT_FOLDER / "monthly_income_histogram.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()
    else:
        print("\nMonthlyIncome column not found in dataset.")

    # ==========================================================
    # 3. Completed
    # ==========================================================

    print("\n" + "=" * 80)
    print("DATA UNDERSTANDING COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print("Dataset:", file_path)
    print("Output folder:", OUTPUT_FOLDER)
    print("Original dataset was not modified.")

except FileNotFoundError:
    print(f"Error: File not found.\n{file_path}")

except pd.errors.EmptyDataError:
    print("Error: CSV file is empty.")

except Exception as e:
    print("An unexpected error occurred:")
    print(e)