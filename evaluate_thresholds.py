
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load the saved model and validation data
model = joblib.load(
    Path("models") / "fraudshield_model.joblib"
)

val_df = pd.read_csv(
    Path("data") / "splits" / "validation.csv"
)

X_val = val_df["TEXT"].fillna("").astype(str)
y_val = val_df["LABEL"]

# Find the probability column belonging to Scam
scam_index = list(model.classes_).index("Scam")
scam_probabilities = model.predict_proba(X_val)[:, scam_index]

print("========== THRESHOLD COMPARISON ==========")

for threshold in [0.30, 0.40, 0.50, 0.60]:
    predictions = [
        "Scam" if probability >= threshold
        else "Legitimate"
        for probability in scam_probabilities
    ]

    precision = precision_score(
        y_val,
        predictions,
        pos_label="Scam",
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        pos_label="Scam",
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        pos_label="Scam",
        zero_division=0
    )

    accuracy = accuracy_score(y_val, predictions)

    matrix = confusion_matrix(
        y_val,
        predictions,
        labels=["Legitimate", "Scam"]
    )

    print(f"\nThreshold: {threshold:.2f}")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Scam precision: {precision * 100:.2f}%")
    print(f"Scam recall:    {recall * 100:.2f}%")
    print(f"Scam F1-score:  {f1:.3f}")
    print("Confusion matrix:")
    print(matrix)

# Inspect scams missed at the default 0.50 threshold
missed = (
    (y_val.to_numpy() == "Scam")
    & (scam_probabilities < 0.50)
)

print("\n========== MISSED SCAM EXAMPLES ==========")

missed_indices = [
    i for i, value in enumerate(missed) if value
]

print("Missed scam messages:", len(missed_indices))

for i in missed_indices[:10]:
    print(f"\nEstimated scam score: "
          f"{scam_probabilities[i] * 100:.2f}%")
    print("Message:", X_val.iloc[i][:500])
