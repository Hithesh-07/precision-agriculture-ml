"""
src/prediction.py
=================
Prediction utilities for single-record and batch prediction.
Uses the saved pipeline from models/best_model.pkl.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    check_out_of_training_range,
    load_schema_info,
    prepare_single_record,
)
from src.model_training import load_best_model, load_evaluation

logger = logging.getLogger(__name__)

# ── batch required columns ────────────────────────────────────────────────────
BATCH_REQUIRED_COLUMNS = ALL_FEATURES  # Crop, Season, State, Crop_Year, Area, ...
MAX_BATCH_ROWS = 5000
MAX_FILE_SIZE_MB = 5


def predict_single(input_dict: dict) -> dict:
    """
    Predict yield for a single record.

    Args:
        input_dict: dict with keys matching ALL_FEATURES

    Returns:
        dict with keys: predicted_yield, model_name, out_of_range_warnings
        Raises ValueError on invalid input.
        Raises FileNotFoundError if no trained model exists.
    """
    schema = load_schema_info()
    eval_meta = load_evaluation()

    # Validate and convert input
    X = prepare_single_record(input_dict, schema)

    # Check out-of-range inputs
    warnings = check_out_of_training_range(input_dict, schema)

    # Load and predict
    pipeline = load_best_model()
    y_log = pipeline.predict(X)
    y_pred = float(np.expm1(np.clip(y_log[0], -1, 20)))

    return {
        "predicted_yield": round(y_pred, 4),
        "model_name": eval_meta.get("best_model", "Unknown"),
        "out_of_range_warnings": warnings,
    }


def validate_batch_file(file_obj, file_name: str, file_size_bytes: int) -> tuple[pd.DataFrame | None, str | None]:
    """
    Validate an uploaded batch file.
    Returns (dataframe, None) on success, or (None, error_message) on failure.
    """
    # File extension check
    if not file_name.lower().endswith(".csv"):
        return None, "File must have a .csv extension."

    # File size check
    if file_size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
        return None, f"File size ({file_size_bytes / 1024 / 1024:.1f} MB) exceeds the {MAX_FILE_SIZE_MB} MB limit."

    # Parse CSV
    try:
        df = pd.read_csv(file_obj)
    except Exception as e:
        return None, f"Could not parse CSV: {e}"

    # Row limit
    if len(df) > MAX_BATCH_ROWS:
        return None, f"File has {len(df)} rows, exceeding the limit of {MAX_BATCH_ROWS}."

    return df, None


def validate_batch_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Returns (present_required_cols, missing_required_cols).
    """
    present = [c for c in BATCH_REQUIRED_COLUMNS if c in df.columns]
    missing = [c for c in BATCH_REQUIRED_COLUMNS if c not in df.columns]
    return present, missing


def predict_batch(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """
    Run predictions on a batch DataFrame.
    Rows with invalid data are skipped and marked.
    Returns a copy of df with added columns: Predicted_Yield, Prediction_Status, Prediction_Error.
    """
    pipeline = load_best_model()
    results = []

    for idx, row in df.iterrows():
        input_dict = row.to_dict()
        try:
            # Strip whitespace from categorical fields
            for col in CATEGORICAL_FEATURES:
                if col in input_dict:
                    input_dict[col] = str(input_dict[col]).strip()

            X = prepare_single_record(input_dict, schema)
            y_log = pipeline.predict(X)
            y_pred = float(np.expm1(np.clip(y_log[0], -1, 20)))
            results.append({
                "Predicted_Yield": round(y_pred, 4),
                "Prediction_Status": "OK",
                "Prediction_Error": "",
            })
        except ValueError as e:
            results.append({
                "Predicted_Yield": None,
                "Prediction_Status": "ERROR",
                "Prediction_Error": str(e),
            })

    result_df = df.copy()
    result_meta = pd.DataFrame(results, index=df.index)
    result_df["Predicted_Yield"] = result_meta["Predicted_Yield"]
    result_df["Prediction_Status"] = result_meta["Prediction_Status"]
    result_df["Prediction_Error"] = result_meta["Prediction_Error"]

    return result_df
