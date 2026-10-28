
import pandas as pd
from pathlib import Path

# Step 1: Define the dataset path
dataset_path = Path("data") / "Dataset_5971.csv"

# Step 2: Check whether the dataset exists
if not dataset_path.exists():
    print("Dataset not found!")
    print("Expected location:", dataset_path.resolve())
    raise SystemExit(1)

# Step 3: Load the dataset
df = pd.read_csv(dataset_path)

# Step 4: Display basic information
print("\n========== DATASET OVERVIEW ==========")

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

print("\n========== DATASET INFORMATION ==========")

df.info()

print("\n========== MISSING VALUES ==========")

print(df.isnull().sum())

print("\n========== DUPLICATE RECORDS ==========")

print("Total duplicate rows:", df.duplicated().sum())

print("\n========== POSSIBLE LABEL COLUMNS ==========")

for column in df.columns:
    if df[column].nunique(dropna=True) <= 20:
        print(f"\nColumn: {column}")
        print(df[column].value_counts(dropna=False).head(20))
