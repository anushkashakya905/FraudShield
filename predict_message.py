import joblib
from pathlib import Path

model_path = Path("models") / "fraudshield_model.joblib"
model = joblib.load(model_path)

SCAM_THRESHOLD = 0.30

print("=" * 45)
print("       FRAUDSHIELD MESSAGE DETECTOR")
print("=" * 45)

while True:
    message = input("\nEnter a message (or type 'exit' to quit): ").strip()

    if message.lower() == "exit":
        print("FraudShield closed.")
        break

    if not message:
        print("Please enter a message.")
        continue

    probabilities = model.predict_proba([message])[0]
    scam_index = list(model.classes_).index("Scam")
    scam_score = probabilities[scam_index]

    if scam_score >= SCAM_THRESHOLD:
        prediction = "Scam"
        explanation = "The message contains patterns associated with scam-labelled messages."
    else:
        prediction = "Legitimate"
        explanation = "The message has fewer patterns associated with scam-labelled messages."

    print("\n---------- ANALYSIS RESULT ----------")
    print("Prediction:", prediction)
    print(f"Estimated scam score: {scam_score * 100:.2f}%")
    print("Explanation:", explanation)
    print("-------------------------------------")
    print("Note: This is a model estimate, not a guarantee of safety.")