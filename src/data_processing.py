"""
src/data_processing.py
======================
Data loading, cleaning, and preprocessing utilities.
All preprocessing is fitted on training data only and applied consistently
to test data and new predictions.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "crop_yield.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"

# ── Column schema ─────────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = ["Crop", "Season", "State"]
NUMERIC_FEATURES = ["Crop_Year", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "Yield"

# Production is excluded because Yield is derived from Production/Area
# (or closely related in many rows), making Production a leaky predictor.
EXCLUDED_FEATURES = ["Production"]

# Reasonable numeric ranges for input validation
FEATURE_RANGES = {
    "Crop_Year":        (1990, 2030),
    "Area":             (0.0001, 1e8),
    "Annual_Rainfall":  (0, 10_000),
    "Fertilizer":       (0, 1e11),
    "Pesticide":        (0, 1e8),
}


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Load the raw CSV dataset without modification."""
    csv_path = path or DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows, %d columns from %s", len(df), df.shape[1], csv_path)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing."""
    required = ALL_FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def clean_data(df: pd.DataFrame, log_decisions: bool = True) -> pd.DataFrame:
    """
    Apply documented, conservative cleaning rules.
    Does NOT silently drop rows to improve metrics.

    Decisions:
    - Strip whitespace from categorical columns.
    - Rows where Area <= 0 are invalid (cannot compute yield per hectare).
    - Rows where Yield < 0 are physically impossible.
    - Rows where Yield == 0 AND Production == 0 are kept but flagged;
      they represent genuine zero-production events.
    - Extreme outliers (Yield > 99.9th percentile) are kept.
      The model is trained on log-transformed yield, which compresses the scale.
    """
    original_len = len(df)

    # Strip whitespace from categorical columns
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].str.strip()

    # Remove rows with non-positive area (physically meaningless)
    before = len(df)
    df = df[df["Area"] > 0].copy()
    if log_decisions:
        logger.info("Removed %d rows with Area <= 0", before - len(df))

    # Remove rows with negative yield
    before = len(df)
    df = df[df[TARGET] >= 0].copy()
    if log_decisions:
        logger.info("Removed %d rows with Yield < 0", before - len(df))

    # Fill any missing values (none expected, but guard anyway)
    for col in NUMERIC_FEATURES:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.warning("Filled missing values in %s with median %.4f", col, median_val)

    final_len = len(df)
    if log_decisions:
        logger.info("Clean data: %d rows kept of %d (%d removed total)",
                    final_len, original_len, original_len - final_len)

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer pipeline.
    - Categorical: OrdinalEncoder (works with tree-based models)
    - Numeric: StandardScaler (helps Linear Regression; harmless for trees)
    """
    cat_pipeline = Pipeline([
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        ))
    ])
    num_pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
            ("num", num_pipeline, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def load_schema_info() -> dict:
    """Load saved schema info (categories, ranges) from outputs directory."""
    schema_path = OUTPUTS_DIR / "feature_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(
            "Feature schema not found. Run train_model.py first."
        )
    with open(schema_path) as f:
        return json.load(f)


def save_schema_info(df: pd.DataFrame) -> None:
    """Save the unique categorical values and numeric ranges from training data."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    schema = {
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target": TARGET,
        "categories": {
            col: sorted(df[col].unique().tolist())
            for col in CATEGORICAL_FEATURES
        },
        "numeric_stats": {
            col: {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "p05": float(df[col].quantile(0.05)),
                "p95": float(df[col].quantile(0.95)),
            }
            for col in NUMERIC_FEATURES
        },
    }
    with open(OUTPUTS_DIR / "feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    logger.info("Feature schema saved.")


def prepare_single_record(input_dict: dict, schema: dict) -> pd.DataFrame:
    """
    Convert a user-supplied input dict to a DataFrame ready for prediction.
    Validates types and categorical membership.
    Raises ValueError with a helpful message on invalid input.
    """
    errors = []

    # Categorical validation
    for col in schema["categorical_features"]:
        val = input_dict.get(col)
        if val is None or str(val).strip() == "":
            errors.append(f"'{col}' is required.")
        elif str(val).strip() not in schema["categories"][col]:
            errors.append(
                f"'{val}' is not a recognised value for {col}. "
                f"Valid options: {schema['categories'][col][:5]}..."
            )

    # Numeric validation
    for col in schema["numeric_features"]:
        val = input_dict.get(col)
        if val is None or (isinstance(val, str) and val.strip() == ""):
            errors.append(f"'{col}' is required.")
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            errors.append(f"'{col}' must be a valid number, got: {val!r}")
            continue

        if np.isnan(fval) or np.isinf(fval):
            errors.append(f"'{col}' cannot be NaN or infinite.")
            continue

        lo, hi = FEATURE_RANGES.get(col, (None, None))
        if lo is not None and fval < lo:
            errors.append(f"'{col}' = {fval} is below the valid minimum ({lo}).")
        if hi is not None and fval > hi:
            errors.append(f"'{col}' = {fval} exceeds the valid maximum ({hi}).")

    if errors:
        raise ValueError("Input validation failed:\n" + "\n".join(f"  • {e}" for e in errors))

    row = {col: [str(input_dict[col]).strip()] for col in schema["categorical_features"]}
    row.update({col: [float(input_dict[col])] for col in schema["numeric_features"]})
    return pd.DataFrame(row)


def check_out_of_training_range(input_dict: dict, schema: dict) -> list[str]:
    """Return a list of warnings for numeric values outside the training data range."""
    warnings = []
    for col in schema["numeric_features"]:
        val = input_dict.get(col)
        if val is None:
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        stats = schema["numeric_stats"].get(col, {})
        p05 = stats.get("p05")
        p95 = stats.get("p95")
        if p05 is not None and fval < p05:
            warnings.append(f"{col} ({fval:.2f}) is below the 5th percentile of training data ({p05:.2f}).")
        elif p95 is not None and fval > p95:
            warnings.append(f"{col} ({fval:.2f}) is above the 95th percentile of training data ({p95:.2f}).")
    return warnings
