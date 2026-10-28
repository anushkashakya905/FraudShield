
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load the trained model
model = joblib.load(
    Path("models") / "fraudshield_model.joblib"
)

# Load the untouched test dataset
test_df = pd.read_csv(
    Path("data") / "splits" / "test.csv"
)

X_test = test_df["TEXT"].fillna("").astype(str)
y_test = test_df["LABEL"]

# Provisional threshold selected using validation data
threshold = 0.30

# Get the model's estimated Scam probability
scam_index = list(model.classes_).index("Scam")
scam_scores = model.predict_proba(X_test)[:, scam_index]

# Apply the selected threshold
predictions = [
    "Scam" if score >= threshold else "Legitimate"
    for score in scam_scores
]

print("========== FINAL TEST RESULTS ==========")
print("Test messages:", len(X_test))
print("Decision threshold:", threshold)

print(
    "\nAccuracy:",
    round(accuracy_score(y_test, predictions) * 100, 2),
    "%"
)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=["Legitimate", "Scam"],
        zero_division=0
    )
)

print("\nConfusion matrix:")
print("Label order: Legitimate, Scam")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=["Legitimate", "Scam"]
    )
)

print("\n========== SAMPLE PREDICTIONS ==========")

for message, score, prediction in zip(
    X_test.head(10),
    scam_scores[:10],
    predictions[:10]
):
    print("\nMessage:", message[:180])
    print("Predicted class:", prediction)
    print(f"Estimated scam score: {score * 100:.2f}%")
