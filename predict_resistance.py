import os
import numpy as np
import rasterio
import joblib
from xgboost import XGBClassifier  # type: ignore
from typing import List, Tuple
import warnings

DATA_DIR = "./data/features"
MODEL_PATH = "./models/run_0/xgb_model.pkl"
OUTPUT_DIR = "./outputs"
OUTPUT_FILENAME = "resistance.tif"

FEATURES: List[str] = [
    "DEM.tif",
    "slope1.tif",
    "road20.tif",
    "ndvi20.tif",
    "pop20.tif",
    "nightlight20.tif",
    "water.tif",
]


def load_model_and_scaler(model_path: str) -> Tuple[XGBClassifier, object, List[str]]:
    data = joblib.load(model_path)
    model: XGBClassifier = data["model"]
    scaler = data["scaler"]
    feature_cols: List[str] = data["feature_cols"]
    print(f"Loaded model from: {model_path}")
    return model, scaler, feature_cols


def read_feature_data(data_dir: str, features: List[str]):
    feature_arrays = []
    nodata_vals = []
    meta = None

    for fname in features:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature file not found: {path}")

        with rasterio.open(path) as src:
            if meta is None:
                meta = src.meta.copy()
            arr = src.read(1)
            feature_arrays.append(arr)
            nodata_vals.append(src.nodata)

    first_shape = feature_arrays[0].shape
    for i, arr in enumerate(feature_arrays[1:]):
        if arr.shape != first_shape:
            raise ValueError(
                f"Shape mismatch between {features[0]} and {features[i + 1]}"
            )

    print(f"Loaded {len(features)} feature rasters, shape: {first_shape}")
    return feature_arrays, nodata_vals, meta


def preprocess_features(
    feature_arrays: List[np.ndarray],
    nodata_vals: List[float],
    scaler,
) -> np.ndarray:
    flattened = [arr.flatten() for arr in feature_arrays]
    X = np.column_stack(flattened)

    for i, nodata in enumerate(nodata_vals):
        if nodata is not None:
            median_value = scaler.mean_[i]
            X[:, i] = np.where(X[:, i] == nodata, median_value, X[:, i])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )
        X_scaled = scaler.transform(X)

    print(f"Preprocessed features, shape: {X_scaled.shape}")
    return X_scaled


def predict_resistance_surface(
    model: XGBClassifier, X_scaled: np.ndarray, meta: dict
) -> np.ndarray:
    y_proba = model.predict_proba(X_scaled)[:, 1]
    if not (np.all(y_proba >= 0) and np.all(y_proba <= 1)):
        raise ValueError("Predicted probabilities are not in [0, 1].")
    resistance_surface = y_proba.reshape(meta["height"], meta["width"])
    print(f"Predicted resistance surface, shape: {resistance_surface.shape}")
    return resistance_surface


def save_resistance_surface(
    output_dir: str,
    filename: str,
    resistance_surface: np.ndarray,
    meta: dict,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    meta_updated = meta.copy()
    meta_updated.update(dtype=rasterio.float32, count=1, nodata=-9999)

    output_path = os.path.join(output_dir, filename)
    with rasterio.open(output_path, "w", **meta_updated) as dst:
        dst.write(resistance_surface.astype(rasterio.float32), 1)

    print(f"Resistance surface saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    try:
        model, scaler, feature_cols = load_model_and_scaler(MODEL_PATH)

        print("\nChecking feature order:")
        print("Train:", feature_cols)
        print("Predict:", [f.split(".")[0] for f in FEATURES])
        if len(feature_cols) != len(FEATURES):
            print("Warning: feature count mismatch.")

        feature_arrays, nodata_vals, meta = read_feature_data(DATA_DIR, FEATURES)

        X_scaled = preprocess_features(feature_arrays, nodata_vals, scaler)

        resistance_surface = predict_resistance_surface(model, X_scaled, meta)

        save_resistance_surface(OUTPUT_DIR, OUTPUT_FILENAME, resistance_surface, meta)

        print("\nDone.")
    except Exception as e:
        print(f"\nError: {e}")
