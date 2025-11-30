import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
)

from xgboost import XGBClassifier
import joblib

DATA_PATH = "./data/samples/sample1.csv"
OUTPUT_DIR = "./models/run_0"

FEATURE_COLS = [
    "DEM",
    "Slope",
    "RoadDist",
    "NDVI",
    "PopDensity",
    "Nightlight",
    "WaterDist",
]
LABEL_COL = "Label"

RANDOM_SEED =
TRAINING_NOISE =


def create_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    print(f"Output directory: {path}")


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    print(f"Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    if df.isnull().any().any():
        print("Handling missing values...")
        fill_values = {}
        for col in df.columns:
            if df[col].dtype in ["float64", "int64"]:
                fill_values[col] = df[col].median()
            else:
                if not df[col].mode().empty:
                    fill_values[col] = df[col].mode()[0]
        df = df.fillna(fill_values)
        print("Missing values handled")
    return df


def standardize_features(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print("Feature standardization complete")
    return df, scaler


def add_noise_to_data(
    X: pd.DataFrame, feature_stds: np.ndarray, noise_factor: float
) -> pd.DataFrame:
    X_noisy = X.copy().values
    for i in range(X.shape[1]):
        noise = noise_factor * feature_stds[i] * np.random.randn(X.shape[0])
        X_noisy[:, i] += noise
    return pd.DataFrame(X_noisy, columns=X.columns)


if __name__ == "__main__":
    create_output_dir(OUTPUT_DIR)

    df = load_data(DATA_PATH)
    df = handle_missing_values(df)
    df, scaler = standardize_features(df, FEATURE_COLS)

    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in data")

    X = df[FEATURE_COLS]
    y = df[LABEL_COL]
    print(f"Feature shape: {X.shape}, label shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"Train set: {X_train.shape}, test set: {X_test.shape}")

    if not X_train.index.intersection(X_test.index).empty:
        print("Warning: overlapping indices between train and test. Re-splitting...")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=RANDOM_SEED,
            stratify=y,
        )

    pos_samples = np.sum(y_train == 1)
    neg_samples = np.sum(y_train == 0)
    scale_pos_weight = neg_samples / pos_samples if pos_samples > 0 else 1.0
    print(f"Class ratio (neg:pos): {scale_pos_weight:.2f}:1")

    print("\nTraining XGBoost model with data augmentation...")

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=1800,
        max_depth=5,
        reg_lambda=7.0,
        reg_alpha=4.5,
        gamma=1.9,
        learning_rate=0.015,
        subsample=0.7,
        colsample_bytree=0.7,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        tree_method="hist",
        use_label_encoder=False,
    )

    feature_stds = np.std(X_train, axis=0).values
    X_train_noisy = add_noise_to_data(X_train, feature_stds, TRAINING_NOISE)

    X_train_combined = pd.concat([X_train, X_train_noisy], axis=0)
    y_train_combined = pd.concat([y_train, y_train], axis=0)

    model.fit(X_train_combined, y_train_combined)
    print(f"Data augmentation used with noise factor {TRAINING_NOISE:.3f}")

    model_path = os.path.join(OUTPUT_DIR, "xgb_model.pkl")
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_cols": FEATURE_COLS,
        },
        model_path,
    )
    print(f"Model saved to: {model_path}")

    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"\nScaler saved to: {scaler_path}")
    print("Scaler means:", scaler.mean_.round(4).tolist())
    print("Scaler stds:", scaler.scale_.round(4).tolist())

    print("\nRunning cross-validation...")

    cv_model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=1800,
        max_depth=5,
        reg_lambda=7.0,
        reg_alpha=4.5,
        gamma=1.9,
        learning_rate=0.018,
        subsample=0.7,
        colsample_bytree=0.7,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        tree_method="hist",
        use_label_encoder=False,
    )

    cv_scores = cross_val_score(
        cv_model,
        X_train_combined,
        y_train_combined,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )

    print("Cross-validation AUC scores:", [round(s, 4) for s in cv_scores])
    print(f"Mean AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    cv_df = pd.DataFrame(
        {"fold": range(1, len(cv_scores) + 1), "auc_score": cv_scores}
    )
    cv_path = os.path.join(OUTPUT_DIR, "cross_validation.csv")
    cv_df.to_csv(cv_path, index=False)
    print(f"Cross-validation results saved to: {cv_path}")

    print("\nEvaluating model...")

    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]

    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Train": {
            "Accuracy": accuracy_score(y_train, y_train_pred),
            "Precision": precision_score(y_train, y_train_pred),
            "Recall": recall_score(y_train, y_train_pred),
            "F1-score": f1_score(y_train, y_train_pred),
            "ROC AUC": roc_auc_score(y_train, y_train_proba),
        },
        "Test": {
            "Accuracy": accuracy_score(y_test, y_test_pred),
            "Precision": precision_score(y_test, y_test_pred),
            "Recall": recall_score(y_test, y_test_pred),
            "F1-score": f1_score(y_test, y_test_pred),
            "ROC AUC": roc_auc_score(y_test, y_test_proba),
        },
    }

    metrics_df = pd.DataFrame(metrics).T.round(4)
    metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.csv")
    metrics_df.to_csv(metrics_path)
    print(f"Model metrics saved to: {metrics_path}")
    print("Metrics summary:")
    print(metrics_df)

    print("\n=== Done ===")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Metrics: {metrics_path}")
    print(f"Cross-validation: {cv_path}")
    print(f"Model file: {model_path}")
    print(f"Scaler file: {scaler_path}")
