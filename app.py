
from pathlib import Path
import math

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__, static_folder="public", static_url_path="")

# Load the trained model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "heart_disease_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

ALLOWED_VALUES = {
    "sex": [0, 1],
    "cp": [1, 2, 3, 4],
    "fbs": [0, 1],
    "restecg": [0, 1, 2],
    "exang": [0, 1],
    "slope": [1, 2, 3],
    "ca": [0, 1, 2, 3],
    "thal": [3, 6, 7]
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({"error": "Invalid request data."}), 400

        # Check that all required measurements are provided
        missing_features = [
            feature for feature in FEATURES if feature not in data
        ]

        if missing_features:
            return jsonify({
                "error": "Please provide all 13 patient measurements."
            }), 400

        patient_data = {}

        # Convert inputs to numbers and reject invalid values
        for feature in FEATURES:
            try:
                value = float(data[feature])
            except (ValueError, TypeError):
                return jsonify({
                    "error": f"Please enter a valid number for {feature}."
                }), 400

            if not math.isfinite(value):
                return jsonify({
                    "error": f"Invalid measurement for {feature}."
                }), 400

            patient_data[feature] = value

        # Validate patient measurements
        if not 1 <= patient_data["age"] <= 120:
            return jsonify({
                "error": "Age must be between 1 and 120."
            }), 400

        # Validate categorical measurements
        for feature, allowed in ALLOWED_VALUES.items():
            if patient_data[feature] not in allowed:
                return jsonify({
                    "error": f"Invalid value for {feature}."
                }), 400

        # Validate resting blood pressure
        if not 50 <= patient_data["trestbps"] <= 300:
            return jsonify({
                "error": "Resting blood pressure must be between 50 and 300 mm Hg."
            }), 400

        # Validate cholesterol
        if not 50 <= patient_data["chol"] <= 700:
            return jsonify({
                "error": "Cholesterol must be between 50 and 700 mg/dl."
            }), 400

        # Validate maximum heart rate
        if not 30 <= patient_data["thalach"] <= 250:
            return jsonify({
                "error": "Maximum heart rate must be between 30 and 250."
            }), 400

        # Validate ST depression
        if not 0 <= patient_data["oldpeak"] <= 10:
            return jsonify({
                "error": "Oldpeak must be between 0 and 10."
            }), 400

        # Prepare patient data in the expected feature order
        patient = pd.DataFrame([patient_data], columns=FEATURES)

        # Generate prediction and probabilities
        prediction = int(model.predict(patient)[0])
        probabilities = model.predict_proba(patient)[0]

        result = {
            "prediction": prediction,
            "result": (
                "Heart disease detected"
                if prediction == 1
                else "No heart disease detected"
            ),
            "confidence": round(
                float(max(probabilities)) * 100, 2
            ),
            "probabilities": {
                "no_disease": round(
                    float(probabilities[0]) * 100, 2
                ),
                "heart_disease": round(
                    float(probabilities[1]) * 100, 2
                )
            }
        }

        return jsonify(result)

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "An error occurred while making the prediction."
        }), 500


if __name__ == "__main__":
    app.run(debug=False)
