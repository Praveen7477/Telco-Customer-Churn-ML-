"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load MLflow-logged model and feature metadata from training
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions to user-friendly output

CRITICAL PATTERN: Training/Serving Consistency
- Uses fixed BINARY_MAP for deterministic binary encoding
- Applies same one-hot encoding with drop_first=True
- Maintains exact feature column order from training
- Handles missing/new categorical values gracefully

Production Deployment:
- MODEL_DIR points to containerized model artifacts
- Feature schema loaded from training-time artifacts
- Optimized for single-row inference (real-time serving)
"""

import glob
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTAINER_MODEL_DIR = Path("/app/model")


# === MODEL LOADING CONFIGURATION ===
def _candidate_model_dirs() -> list[str]:
    candidates: list[str] = []

    env_model_path = os.getenv("MODEL_PATH")
    if env_model_path:
        candidates.append(env_model_path)

    if DEFAULT_CONTAINER_MODEL_DIR.exists():
        candidates.append(str(DEFAULT_CONTAINER_MODEL_DIR))

    local_serving_models = sorted(
        glob.glob(str(PROJECT_ROOT / "src" / "serving" / "model" / "*" / "artifacts" / "model")),
        key=os.path.getmtime,
        reverse=True,
    )
    candidates.extend(local_serving_models)

    local_mlruns_models = sorted(
        glob.glob(str(PROJECT_ROOT / "mlruns" / "*" / "*" / "artifacts" / "model")),
        key=os.path.getmtime,
        reverse=True,
    )
    candidates.extend(local_mlruns_models)

    # Deduplicate while preserving priority order.
    seen = set()
    ordered = []
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if normalized not in seen:
            seen.add(normalized)
            ordered.append(candidate)
    return ordered


def _load_model() -> tuple[object, str]:
    errors = []
    for candidate in _candidate_model_dirs():
        try:
            loaded_model = mlflow.pyfunc.load_model(candidate)
            print(f"Loaded model successfully from {candidate}")
            return loaded_model, candidate
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    raise RuntimeError(
        "Failed to load model from any known location. Checked: " + " | ".join(errors)
    )


model, MODEL_DIR = _load_model()


# === FEATURE SCHEMA LOADING ===
def _load_feature_columns(model_dir: str) -> list[str]:
    model_path = Path(model_dir)
    candidate_files = []

    env_feature_path = os.getenv("FEATURE_COLUMNS_PATH")
    if env_feature_path:
        candidate_files.append(Path(env_feature_path))

    candidate_files.extend(
        [
            model_path / "feature_columns.txt",
            model_path.parent / "feature_columns.txt",
            model_path / "preprocessing.pkl",
            model_path.parent / "preprocessing.pkl",
            PROJECT_ROOT / "artifacts" / "feature_columns.json",
            PROJECT_ROOT / "artifacts" / "preprocessing.pkl",
        ]
    )

    for candidate in candidate_files:
        if not candidate.exists():
            continue

        if candidate.suffix == ".txt":
            with candidate.open() as handle:
                feature_cols = [line.strip() for line in handle if line.strip()]
            if feature_cols:
                print(f"Loaded {len(feature_cols)} feature columns from {candidate}")
                return feature_cols

        if candidate.suffix == ".pkl":
            payload = joblib.load(candidate)
            feature_cols = payload.get("feature_columns") if isinstance(payload, dict) else None
            if feature_cols:
                print(f"Loaded {len(feature_cols)} feature columns from {candidate}")
                return feature_cols

        if candidate.suffix == ".json":
            feature_frame = pd.read_json(candidate, typ="series")
            feature_cols = [str(value) for value in feature_frame.tolist() if str(value).strip()]
            if feature_cols:
                print(f"Loaded {len(feature_cols)} feature columns from {candidate}")
                return feature_cols

    raise RuntimeError(f"Failed to load feature columns for model at {model_dir}")


FEATURE_COLS = _load_feature_columns(MODEL_DIR)


# === FEATURE TRANSFORMATION CONSTANTS ===
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(0)

    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .map(mapping)
                .astype("Int64")
                .fillna(0)
                .astype(int)
            )

    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    return df


def predict(input_dict: dict) -> str:
    """
    Main prediction function for customer churn inference.
    """
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    try:
        preds = model.predict(df_enc)
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
    except Exception as exc:
        raise Exception(f"Model prediction failed: {exc}")

    if result == 1:
        return "Likely to churn"
    return "Not likely to churn"
