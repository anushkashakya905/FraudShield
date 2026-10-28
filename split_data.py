
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Step 1: Load the cleaned dataset
input_path = Path("data") / "fraudshield_clean.csv"
df = pd.read_csv(input_path)

print("Total cleaned messages:", len(df))

# Step 2: Separate message text and labels
X = df["TEXT"]
y = df["LABEL"]

# Step 3: Split into 70% training and 30% temporary data
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Step 4: Divide the temporary data equally
# This gives us 15% validation and 15% testing overall
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Step 5: Create output folder
output_dir = Path("data") / "splits"
output_dir.mkdir(parents=True, exist_ok=True)

# Step 6: Save all three subsets
pd.DataFrame({
    "TEXT": X_train,
    "LABEL": y_train
}).to_csv(output_dir / "train.csv", index=False)

pd.DataFrame({
    "TEXT": X_val,
    "LABEL": y_val
}).to_csv(output_dir / "validation.csv", index=False)

pd.DataFrame({
    "TEXT": X_test,
    "LABEL": y_test
}).to_csv(output_dir / "test.csv", index=False)

# Step 7: Display the results
datasets = {
    "Training": (X_train, y_train),
    "Validation": (X_val, y_val),
    "Testing": (X_test, y_test)
}

print("\n========== DATA SPLIT RESULTS ==========")

for name, (messages, labels) in datasets.items():
    print(f"\n{name} dataset")
    print("Total messages:", len(messages))
    print("Class distribution:")
    print(labels.value_counts())
    print("Class percentages:")
    print(
        (labels.value_counts(normalize=True) * 100).round(2)
    )

print("\nAll datasets saved in:", output_dir.resolve())
