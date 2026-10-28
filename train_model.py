
import pandas as pd
import joblib

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Step 1: Define file paths
data_dir = Path("data") / "splits"
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

# Step 2: Load training and validation data
train_df = pd.read_csv(data_dir / "train.csv")
val_df = pd.read_csv(data_dir / "validation.csv")

X_train = train_df["TEXT"].fillna("").astype(str)
y_train = train_df["LABEL"]

X_val = val_df["TEXT"].fillna("").astype(str)
y_val = val_df["LABEL"]

print("Training messages:", len(X_train))
print("Validation messages:", len(X_val))

# Step 3: Create the complete ML pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

# Step 4: Train the model
print("\nTraining the model...")
model.fit(X_train, y_train)

print("Training completed!")

# Step 5: Predict validation labels
val_predictions = model.predict(X_val)

# Step 6: Evaluate the validation results
print("\n========== VALIDATION RESULTS ==========")

print(
    "Accuracy:",
    round(accuracy_score(y_val, val_predictions) * 100, 2),
    "%"
)

print("\nClassification report:")
print(
    classification_report(
        y_val,
        val_predictions,
        labels=["Legitimate", "Scam"],
        zero_division=0
    )
)

print("\nConfusion matrix:")
print("Label order: Legitimate, Scam")
print(
    confusion_matrix(
        y_val,
        val_predictions,
        labels=["Legitimate", "Scam"]
    )
)

# Step 7: Save the trained pipeline
model_path = model_dir / "fraudshield_model.joblib"
joblib.dump(model, model_path)

print("\nModel saved to:", model_path.resolve())
