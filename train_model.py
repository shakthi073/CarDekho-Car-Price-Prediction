"""Download the referenced CarDekho data, train models, evaluate and export the tuned model."""
from pathlib import Path
import json
import pickle
import urllib.request

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
DATA_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)

DATA_URL = (
    "https://raw.githubusercontent.com/manishkr1754/"
    "CarDekho_Used_Car_Price_Prediction/main/notebooks/data/cardekho_dataset.csv"
)
CSV_PATH = DATA_DIR / "cardekho_dataset.csv"
MODEL_PATH = ROOT / "best_model.pkl"

def download_dataset():
    if not CSV_PATH.exists():
        print("Downloading the dataset...")
        urllib.request.urlretrieve(DATA_URL, CSV_PATH)
    return pd.read_csv(CSV_PATH, index_col=0)

def prepare_data(raw):
    """Follow the notebook's selected features and one-hot encoding approach."""
    required = {
        "car_name", "brand", "model", "seller_type", "fuel_type",
        "transmission_type", "vehicle_age", "km_driven", "mileage",
        "engine", "max_power", "seats", "selling_price"
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    data = raw.copy()
    data = data.drop_duplicates()
    for col in ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats", "selling_price"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=list(required))
    data = data[(data["selling_price"] > 0) & (data["vehicle_age"] >= 0)]
    data = data.drop(columns=["car_name", "brand", "model"])

    X_raw = data.drop(columns=["selling_price"])
    y = data["selling_price"].astype(float)
    X = pd.get_dummies(X_raw, dtype=float)

    # Reindex to the exact columns expected by the original Flask interface/model.
    expected = [
        "vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats",
        "seller_type_Dealer", "seller_type_Individual", "seller_type_Trustmark Dealer",
        "fuel_type_CNG", "fuel_type_Diesel", "fuel_type_Electric", "fuel_type_LPG",
        "fuel_type_Petrol", "transmission_type_Automatic", "transmission_type_Manual",
    ]
    X = X.reindex(columns=expected, fill_value=0)
    return X, y

def main():
    raw = download_dataset()
    X, y = prepare_data(raw)
    print(f"Prepared dataset: {X.shape[0]:,} rows × {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = [
        ("Linear Regression", LinearRegression()),
        ("Support Vector Regression", SVR()),
        ("Decision Tree", DecisionTreeRegressor(random_state=42)),
        ("Random Forest", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ("Ridge", Ridge()),
        ("Lasso", Lasso()),
    ]
    rows = []
    for name, estimator in candidates:
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "R2": r2_score(y_test, pred),
            "Explained Variance": explained_variance_score(y_test, pred),
        })
    comparison = pd.DataFrame(rows).sort_values("RMSE")
    comparison.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)
    print("\nBaseline model comparison:")
    print(comparison.to_string(index=False))

    params = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 10, 20, 30, 40, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [1.0, "sqrt", "log2"],
        "bootstrap": [True, False],
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=params,
        n_iter=100,
        cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    pred = best_model.predict(X_test)

    metrics = {
        "model": "RandomForestRegressor (RandomizedSearchCV)",
        "rows_after_preprocessing": int(len(X)),
        "features": list(X.columns),
        "test_size": 0.2,
        "random_state": 42,
        "best_parameters": search.best_params_,
        "MAE": float(mean_absolute_error(y_test, pred)),
        "MSE": float(mean_squared_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred)),
        "Explained_Variance": float(explained_variance_score(y_test, pred)),
    }
    with MODEL_PATH.open("wb") as f:
        pickle.dump(best_model, f)
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nTuned model evaluation:")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model: {MODEL_PATH}")

if __name__ == "__main__":
    main()
