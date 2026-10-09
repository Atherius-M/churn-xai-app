import json
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def export_model(
    model_name: str,
    pipeline,
    feature_cols: list,
    numeric_cols: list,
    categorical_cols: list,
    X_background: pd.DataFrame,
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
    target_col: str,
    threshold: float,
) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "model.joblib")
    X_background.to_parquet(MODEL_DIR / "background.parquet")
    X_sample.to_parquet(MODEL_DIR / "sample.parquet")
    y_sample.to_frame("target").to_parquet(MODEL_DIR / "sample_target.parquet")

    metadata = {
        "model_name": model_name,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "target_col": target_col,
        "threshold": threshold,
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return MODEL_DIR


def model_exists() -> bool:
    return (MODEL_DIR / "model.joblib").exists()


def load_exported_model():
    pipeline = joblib.load(MODEL_DIR / "model.joblib")
    with open(MODEL_DIR / "metadata.json") as f:
        metadata = json.load(f)
    X_background = pd.read_parquet(MODEL_DIR / "background.parquet")
    X_sample = pd.read_parquet(MODEL_DIR / "sample.parquet")
    y_sample = pd.read_parquet(MODEL_DIR / "sample_target.parquet")["target"]
    return pipeline, metadata, X_background, X_sample, y_sample
