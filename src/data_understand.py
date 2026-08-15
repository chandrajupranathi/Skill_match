import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("1. Load the Dataset")
print("-----------------------------------")


# Get project folder automatically
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Dataset path
file_path = PROJECT_DIR / "dataset" / "reordered_raw.csv"


try:

    # Read CSV file
    df = pd.read_csv(file_path)


    # ========================================================
    # 2. DISPLAY DATASET
    # ========================================================

    print("-----------------------------------")
    print("1. Dataset Contents:")
    print("-----------------------------------")

    print(df)


    # ========================================================
    # 3. NUMBER OF ROWS AND COLUMNS
    # ========================================================

    print("-----------------------------------")
    print("\n2. Number of Rows and Columns:")
    print("-----------------------------------")

    print(df.shape)


    # ========================================================
    # 4. COLUMN NAMES
    # ========================================================

    print("\n3. Column Names:")
    print("-----------------------------------")

    print(df.columns.tolist())


    # Configure pandas display
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)


    # ========================================================
    # 5. FIRST 10 RECORDS
    # ========================================================

    print("-----------------------------------")
    print("\n4. Dataset First 10 Records")
    print("-----------------------------------")

    print(df.head(10))


    # ========================================================
    # 6. LAST 10 RECORDS
    # ========================================================

    print("****************************************************************")

    print("Dataset Last 10 Records")

    print("-----------------------------------")

    print(df.tail(10))


    # ========================================================
    # 7. UNDERSTAND DATASET
    # ========================================================

    print("-----------------------------------")
    print("2. Understand the Dataset")
    print("-----------------------------------")


    # ========================================================
    # 8. DATA TYPES
    # ========================================================

    print("-----------------------------------")
    print("\n5. Data Types of Columns:")

    print(df.dtypes)


    # ========================================================
    # 9. COLUMN NAME + DATA TYPE
    # ========================================================

    print("=" * 60)

    print("6. Display Column Names with Data Types")

    print("\nColumn Name\t\tData Type")

    print("-" * 40)

    for column in df.columns:

        print(
            f"{column:<30} {df[column].dtype}"
        )


    # ========================================================
    # 10. DATASET INFORMATION
    # ========================================================

    print("-----------------------------------")
    print("7. Dataset Summary and Information")
    print("-----------------------------------")

    df.info()

    print("\n" + "=" * 50 + "\n")


    # ========================================================
    # 11. NUMERICAL COLUMNS
    # ========================================================

    print("8. Display Numeric Columns")

    print("-----------------------------------")

    # Select numerical columns
    numeric_df = df.select_dtypes(
        include=["number"]
    )


    print("Numerical Columns:")

    print(numeric_df.columns.tolist())


    # ========================================================
    # 12. MISSING VALUES IN NUMERIC COLUMNS
    # ========================================================

    print("\n9. Missing Values in Numeric Attributes")

    print(numeric_df.isnull().sum())


    print(
        "\n10. Total Missing Numeric Values:",
        numeric_df.isnull().sum().sum()
    )


    # ========================================================
    # 13. FLOAT COLUMNS
    # ========================================================

    float_columns = df.select_dtypes(
        include=["float"]
    ).columns


    print("-----------------------------------")

    print("11. Float Attribute Names:")

    print("-----------------------------------")


    for column in float_columns:

        print(column)


    # ========================================================
    # 14. MISSING VALUES IN FLOAT COLUMNS
    # ========================================================

    print("12. Missing Values in Float Attributes")

    print("=" * 50)


    # Correct way:
    print(
        df[float_columns].isnull().sum()
    )


    print(
        "\nTotal Missing Float Values:",
        df[float_columns].isnull().sum().sum()
    )


    # ========================================================
    # 15. CATEGORICAL COLUMNS
    # ========================================================

    print("-----------------------------------")

    print("13. Display Categorical Attributes:")

    print("-----------------------------------")


    # Your dataset contains these categorical columns
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


    print(categorical_columns)


    # ========================================================
    # 16. DISPLAY CATEGORICAL DATA
    # ========================================================

    categorical_df = df[categorical_columns]

    print("\nCategorical Data:")

    print(categorical_df)


    # ========================================================
    # 17. MISSING VALUES IN CATEGORICAL COLUMNS
    # ========================================================

    print("Missing Values in Categorical Attributes")

    print("=" * 50)

    print(
        categorical_df.isnull().sum()
    )


    print(
        "\n14. Total Missing Categorical Values:",
        categorical_df.isnull().sum().sum()
    )


    # ========================================================
    # 18. MISSING VALUES IN EACH COLUMN
    # ========================================================

    print("-----------------------------------")

    print("15. Missing Values in Each Column")

    print("-" * 40)

    print(df.isnull().sum())


    # ========================================================
    # 19. TOTAL MISSING VALUES
    # ========================================================

    total_missing = df.isnull().sum().sum()

    print("-----------------------------------")

    print("16. Total Missing Values:", total_missing)


    # ========================================================
    # 20. DUPLICATE RECORDS
    # ========================================================

    print("-----------------------------------")

    duplicate_count = df.duplicated().sum()

    print(
        "17. Number of Duplicate Records:",
        duplicate_count
    )


    # ========================================================
    # 21. STATISTICAL OVERVIEW
    # ========================================================

    print("-----------------------------------")

    print("18. Statistical Overview")

    print("-----------------------------------")

    print(df.describe())


    # ========================================================
    # 22. TARGET VARIABLE
    # ========================================================

    print("\n-----------------------------------")

    print("19. Target Variable - Attrition")

    print("-----------------------------------")


    print(
        df["Attrition"].value_counts()
    )


    # ========================================================
    # 23. HISTOGRAM OF AGE
    # ========================================================

    print("\n-----------------------------------")

    print("20. Display Histogram of Age")

    print("-----------------------------------")


    plt.figure(figsize=(8, 5))


    plt.hist(
        df["Age"],
        bins=10,
        edgecolor="black"
    )


    plt.title("Histogram of Age")

    plt.xlabel("Age")

    plt.ylabel("Frequency")

    plt.grid(True)

    plt.show()


    # ========================================================
    # 24. HISTOGRAM OF MONTHLY INCOME
    # ========================================================

    print("\n-----------------------------------")

    print("21. Display Histogram of Monthly Income")

    print("-----------------------------------")


    plt.figure(figsize=(8, 5))


    plt.hist(
        df["MonthlyIncome"],
        bins=10,
        edgecolor="black"
    )


    plt.title("Histogram of Monthly Income")

    plt.xlabel("Monthly Income")

    plt.ylabel("Frequency")

    plt.grid(True)

    plt.show()


    # ========================================================
    # 25. ATTRITION COUNT PLOT
    # ========================================================

    print("\n-----------------------------------")

    print("22. Attrition Distribution")

    print("-----------------------------------")


    plt.figure(figsize=(7, 5))


    sns.countplot(
        data=df,
        x="Attrition"
    )


    plt.title("Employee Attrition Distribution")

    plt.xlabel("Attrition")

    plt.ylabel("Number of Employees")

    plt.show()


    # ========================================================
    # 26. COMPLETED
    # ========================================================

    print("\n" + "=" * 60)

    print("DATA UNDERSTANDING COMPLETED SUCCESSFULLY")

    print("=" * 60)


except FileNotFoundError:

    print(
        f"Error: The file at '{file_path}' was not found."
    )


except Exception as e:

    print(
        f"An error occurred: {e}"
    )