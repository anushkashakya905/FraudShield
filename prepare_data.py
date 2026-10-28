
import pandas as pd
from pathlib import Path

# Step 1: Load the original dataset
input_path = Path("data") / "Dataset_5971.csv"

df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)

# Step 2: Keep only the columns needed for our first model
df = df[["LABEL", "TEXT"]].copy()

# Step 3: Standardize labels
df["LABEL"] = df["LABEL"].str.strip().str.lower()

label_mapping = {
    "ham": "Legitimate",
    "spam": "Scam",
    "smishing": "Scam"
}

df["LABEL"] = df["LABEL"].map(label_mapping)

# Step 4: Remove rows with missing or empty values
df = df.dropna(subset=["LABEL", "TEXT"])

df["TEXT"] = df["TEXT"].str.strip()

df = df[df["TEXT"] != ""]

# Step 5: Check for conflicting labels
label_counts = df.groupby("TEXT")["LABEL"].nunique()

conflicting_texts = label_counts[label_counts > 1].index

print("\nMessages with conflicting labels:", len(conflicting_texts))

# Remove conflicting examples for the initial model
df = df[~df["TEXT"].isin(conflicting_texts)].copy()

# Step 6: Remove exact duplicate records
before = len(df)

df = df.drop_duplicates(subset=["TEXT", "LABEL"])

removed_duplicates = before - len(df)

print("Duplicate records removed:", removed_duplicates)

# Step 7: Save the cleaned dataset
output_path = Path("data") / "fraudshield_clean.csv"

df.to_csv(output_path, index=False)

# Step 8: Display the results
print("\n========== CLEANED DATASET ==========")

print("Dataset shape:", df.shape)

print("\nClass distribution:")
print(df["LABEL"].value_counts())

print("\nClass percentages:")
print((df["LABEL"].value_counts(normalize=True) * 100).round(2))

print("\nSaved to:", output_path.resolve())
