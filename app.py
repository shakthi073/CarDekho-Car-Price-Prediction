"""Flask application for estimating Indian used-car selling prices."""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"

app = Flask(__name__)
app.secret_key = "change-this-secret-key-before-public-deployment"

EXPECTED_COLUMNS = [
    "vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats",
    "seller_type_Dealer", "seller_type_Individual", "seller_type_Trustmark Dealer",
    "fuel_type_CNG", "fuel_type_Diesel", "fuel_type_Electric",
    "fuel_type_LPG", "fuel_type_Petrol",
    "transmission_type_Automatic", "transmission_type_Manual",
]

NUMERIC_FIELDS = {
    "vehicle_age": (0, 50),
    "km_driven": (0, 2_000_000),
    "mileage": (0, 100),
    "engine": (0, 10_000),
    "max_power": (0, 2_000),
    "seats": (1, 20),
}
ALLOWED = {
    "seller_type": {"Dealer", "Individual", "Trustmark Dealer"},
    "fuel_type": {"CNG", "Diesel", "Electric", "LPG", "Petrol"},
    "transmission_type": {"Automatic", "Manual"},
}

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run 'python train_model.py' before starting the app."
        )
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)

model = None
try:
    model = load_model()
except FileNotFoundError:
    # Keep the home page accessible and show setup instructions instead of crashing import.
    pass

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction_text=None, error_text=None)

@app.route("/predict", methods=["POST"])
def predict():
    global model
    try:
        if model is None:
            model = load_model()

        row = {column: 0 for column in EXPECTED_COLUMNS}
        for field, (minimum, maximum) in NUMERIC_FIELDS.items():
            value = float(request.form[field])
            if not np.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{field.replace('_', ' ').title()} must be between {minimum} and {maximum}.")
            row[field] = int(value) if field == "seats" else value

        for field, choices in ALLOWED.items():
            value = request.form[field]
            if value not in choices:
                raise ValueError(f"Invalid {field.replace('_', ' ')} selection.")
            encoded_column = f"{field}_{value}"
            if encoded_column not in row:
                raise ValueError(f"Unsupported {field} value.")
            row[encoded_column] = 1

        input_df = pd.DataFrame([row], columns=EXPECTED_COLUMNS)
        prediction = float(model.predict(input_df)[0])
        if not np.isfinite(prediction):
            raise ValueError("The model returned an invalid prediction.")

        return render_template(
            "index.html",
            prediction_text=f"Estimated selling price: ₹{prediction:,.0f}",
            error_text=None,
            form_data=request.form,
        )
    except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        return render_template(
            "index.html",
            prediction_text=None,
            error_text=str(exc),
            form_data=request.form,
        ), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return render_template(
            "index.html",
            prediction_text=None,
            error_text="Prediction failed. Check the model file and input values.",
            form_data=request.form,
        ), 500

if __name__ == "__main__":
    # Debug is disabled by default. Use a production WSGI server for deployment.
    app.run(host="127.0.0.1", port=5000, debug=False)
