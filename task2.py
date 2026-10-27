
# Prodigy Infotech - Data Science Internship
# Task 02: Data Cleaning and Exploratory Data Analysis (Titanic Dataset)

import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'Segoe UI', 'font.size': 11})

# Create required directories
os.makedirs("output", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

dataset_path = "dataset/train.csv"

# Automatically download dataset if not present
if not os.path.exists(dataset_path):
    print("Dataset not found locally. Downloading from Prodigy InfoTech repository...")
    url = "https://raw.githubusercontent.com/Prodigy-InfoTech/data-science-datasets/main/Task%202/train.csv"
    try:
        urllib.request.urlretrieve(url, dataset_path)
        print(f"Downloaded train.csv to '{dataset_path}'.")
    except Exception as e:
        print(f"Failed to download from GitHub: {e}")
        print("Creating dataset from seaborn titanic fallback...")
        sb_df = sns.load_dataset("titanic")
        # Map seaborn titanic to kaggle titanic format
        fallback_df = pd.DataFrame({
            'PassengerId': range(1, len(sb_df) + 1),
            'Survived': sb_df['survived'],
            'Pclass': sb_df['pclass'],
            'Name': [f"Passenger, {i}" for i in range(1, len(sb_df) + 1)],
            'Sex': sb_df['sex'],
            'Age': sb_df['age'],
            'SibSp': sb_df['sibsp'],
            'Parch': sb_df['parch'],
            'Ticket': [f"TICKET-{i}" for i in range(1, len(sb_df) + 1)],
            'Fare': sb_df['fare'],
            'Cabin': sb_df['deck'].astype(str).replace('nan', np.nan),
            'Embarked': sb_df['embarked']
        })
        fallback_df.to_csv(dataset_path, index=False)
        print(f"Created fallback train.csv at '{dataset_path}'.")

# Load Titanic dataset
print(f"\n{'='*50}\nLOADING DATASET\n{'='*50}")
df = pd.read_csv(dataset_path)

# 1. Explore the dataset
print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n--- Dataset Information ---")
df.info()

print("\n--- Missing Values Before Cleaning ---")
missing_before = df.isnull().sum()
print(missing_before[missing_before > 0])

print(f"\n--- Duplicate Rows ---: {df.duplicated().sum()}")

# 2. Data Cleaning
print(f"\n{'='*50}\nDATA CLEANING\n{'='*50}")
df = df.drop_duplicates().copy()

# Drop Cabin if heavily missing (>70% missing)
if "Cabin" in df.columns:
    missing_cabin_pct = (df["Cabin"].isnull().sum() / len(df)) * 100
    print(f"Dropping 'Cabin' column (missing {missing_cabin_pct:.1f}% of values)")
    df = df.drop(columns=["Cabin"])

# Impute Age with median based on Pclass and Sex for more realistic imputation
if "Age" in df.columns and "Pclass" in df.columns and "Sex" in df.columns:
    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(lambda x: x.fillna(x.median()))
    df["Age"] = df["Age"].fillna(df["Age"].median())
    print("Imputed missing 'Age' using median grouped by Pclass & Sex.")

# Fill Embarked with mode
if "Embarked" in df.columns and df["Embarked"].isnull().any():
    mode_embarked = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(mode_embarked)
    print(f"Imputed missing 'Embarked' with mode ('{mode_embarked}').")

# Fill Fare with median if missing
if "Fare" in df.columns and df["Fare"].isnull().any():
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    print("Imputed missing 'Fare' with median.")

# Create useful feature: FamilySize
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

print("\n--- Missing Values After Cleaning ---")
print(df.isnull().sum())

# 3. Statistical Analysis
print(f"\n{'='*50}\nSTATISTICAL SUMMARY\n{'='*50}")
print(df.describe(include="all"))

# 4. Visualizations & EDA
print(f"\n{'='*50}\nGENERATING VISUALIZATIONS\n{'='*50}")

# Visualization 1: Survival Count
plt.figure(figsize=(7, 5))
ax = sns.countplot(data=df, x="Survived", hue="Survived", palette=["#e74c3c", "#2ecc71"], legend=False)
plt.title("Titanic Survival Count", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("Survived (0 = No, 1 = Yes)", fontsize=11)
plt.ylabel("Number of Passengers", fontsize=11)
plt.xticks([0, 1], ["Did Not Survive (0)", "Survived (1)"])
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig("output/survival_count.png", dpi=300)
plt.close()
print("Saved: output/survival_count.png")

# Visualization 2: Survival by Gender
plt.figure(figsize=(7, 5))
ax = sns.countplot(data=df, x="Sex", hue="Survived", palette=["#e74c3c", "#2ecc71"])
plt.title("Survival by Gender", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("Gender", fontsize=11)
plt.ylabel("Number of Passengers", fontsize=11)
plt.legend(title="Status", labels=["Did Not Survive", "Survived"])
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig("output/survival_by_gender.png", dpi=300)
plt.close()
print("Saved: output/survival_by_gender.png")

# Visualization 3: Survival by Passenger Class
plt.figure(figsize=(7, 5))
ax = sns.countplot(data=df, x="Pclass", hue="Survived", palette=["#e74c3c", "#2ecc71"])
plt.title("Survival by Passenger Class", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("Passenger Class (1st, 2nd, 3rd)", fontsize=11)
plt.ylabel("Number of Passengers", fontsize=11)
plt.legend(title="Status", labels=["Did Not Survive", "Survived"])
plt.tight_layout()
plt.savefig("output/survival_by_class.png", dpi=300)
plt.close()
print("Saved: output/survival_by_class.png")

# Visualization 4: Age Distribution by Survival
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Age", hue="Survived", bins=25, kde=True, palette=["#e74c3c", "#2ecc71"], multiple="stack")
plt.title("Age Distribution by Survival Status", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("Age (years)", fontsize=11)
plt.ylabel("Number of Passengers", fontsize=11)
plt.legend(title="Status", labels=["Survived", "Did Not Survive"])
plt.tight_layout()
plt.savefig("output/age_distribution.png", dpi=300)
plt.close()
print("Saved: output/age_distribution.png")

# Visualization 5: Correlation Heatmap
numeric_cols = df.select_dtypes(include=np.number)
plt.figure(figsize=(9, 6))
sns.heatmap(
    numeric_cols.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={'label': 'Pearson Correlation'}
)
plt.title("Correlation Heatmap of Numerical Features", fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("output/correlation_heatmap.png", dpi=300)
plt.close()
print("Saved: output/correlation_heatmap.png")

# Visualization 6: Fare Distribution by Passenger Class
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Pclass", y="Fare", hue="Pclass", palette="Set2", legend=False)
plt.title("Fare Distribution Across Passenger Classes", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("Passenger Class", fontsize=11)
plt.ylabel("Fare ($)", fontsize=11)
plt.yscale("log")
plt.tight_layout()
plt.savefig("output/fare_by_class.png", dpi=300)
plt.close()
print("Saved: output/fare_by_class.png")

# 5. Key Survival Rate Insights
print(f"\n{'='*50}\nKEY INSIGHTS & SURVIVAL RATES\n{'='*50}")
overall_survival = df["Survived"].mean() * 100
print(f"Overall Survival Rate: {overall_survival:.2f}%")

print("\nSurvival Rate by Gender:")
gender_rates = df.groupby("Sex")["Survived"].agg(["count", "mean"])
gender_rates["mean"] = (gender_rates["mean"] * 100).round(2).astype(str) + "%"
gender_rates.columns = ["Total Passengers", "Survival Rate"]
print(gender_rates)

print("\nSurvival Rate by Passenger Class:")
pclass_rates = df.groupby("Pclass")["Survived"].agg(["count", "mean"])
pclass_rates["mean"] = (pclass_rates["mean"] * 100).round(2).astype(str) + "%"
pclass_rates.columns = ["Total Passengers", "Survival Rate"]
print(pclass_rates)

print("\nSurvival Rate by Embarkation Port:")
embarked_map = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
df['EmbarkedName'] = df['Embarked'].map(embarked_map)
embarked_rates = df.groupby("EmbarkedName")["Survived"].agg(["count", "mean"])
embarked_rates["mean"] = (embarked_rates["mean"] * 100).round(2).astype(str) + "%"
embarked_rates.columns = ["Total Passengers", "Survival Rate"]
print(embarked_rates)

# 6. Save Cleaned Dataset
df_cleaned = df.drop(columns=["EmbarkedName"], errors="ignore")
df_cleaned.to_csv("output/cleaned_titanic.csv", index=False)
print(f"\nCleaned dataset saved to 'output/cleaned_titanic.csv' ({len(df_cleaned)} rows)")

print(f"\n{'='*50}\nTask 02 Completed Successfully!\n{'='*50}")
print("Generated outputs in 'output/':")
for file in sorted(os.listdir("output")):
    print(f" - output/{file}")

