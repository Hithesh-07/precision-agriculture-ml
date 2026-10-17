"""
src/data_processing.py
======================
Data loading, cleaning, and preprocessing utilities.
All preprocessing is fitted on training data only and applied consistently
to test data and new predictions.

Dataset registry
----------------
Four datasets are managed by this project. Only the first two share the full
feature schema required for model training.

  DATA_PATH                   — Original 1997-2020 state-level dataset (training)
  DATA_PATH_MODEL_READY       — Curated 2000-2020 state-level dataset (training)
  DATA_PATH_DISTRICT          — District-level 2000-2020 (analysis only; no rainfall/fertilizer/pesticide)
  DATA_PATH_OFFICIAL_2021_25  — All-India official 2020-21 to 2024-25 (no yield, no area; display only)
  DATA_PATH_COMBINED          — District + All-India combined 2000-2025 (analysis only)

Never combine district and all-India records in the same analysis without
filtering by geography_level. Never use DATA_PATH_DISTRICT,
DATA_PATH_OFFICIAL_2021_25, or DATA_PATH_COMBINED for model training.
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
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "crop_yield.csv"

# New datasets
DATA_PATH_MODEL_READY      = DATA_DIR / "cropzen_model_ready_2000_2020.csv"
DATA_PATH_DISTRICT         = DATA_DIR / "cropzen_india_district_2000_2020.csv"
DATA_PATH_OFFICIAL_2021_25 = DATA_DIR / "cropzen_india_official_all_india_2021_2025.csv"
DATA_PATH_COMBINED         = DATA_DIR / "cropzen_india_combined_2000_2025.csv"

OUTPUTS_DIR = BASE_DIR / "outputs"

# ── Column schema ─────────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = ["Crop", "Season", "State"]
NUMERIC_FEATURES = ["Crop_Year", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "Yield"

# Production is excluded because Yield is derived from or strongly correlated
# with Production/Area. Production is a post-harvest observed quantity not
# available at prediction time, and including it would cause data leakage.
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


# ── New dataset loaders ───────────────────────────────────────────────────────

def load_model_ready_data() -> pd.DataFrame | None:
    """
    Load the curated model-ready dataset (2000–2020, state level).
    Schema is identical to the original crop_yield.csv after whitespace stripping.
    Returns None (not raises) if the file is absent, so the app degrades gracefully.
    """
    if not DATA_PATH_MODEL_READY.exists():
        logger.warning("Model-ready dataset not found at %s", DATA_PATH_MODEL_READY)
        return None
    df = pd.read_csv(DATA_PATH_MODEL_READY, low_memory=False)
    # Strip whitespace from categorical columns (season values have trailing spaces)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].str.strip()
    logger.info("Loaded model-ready dataset: %d rows, %d columns", len(df), df.shape[1])
    return df


def load_district_data() -> pd.DataFrame | None:
    """
    Load the district-level historical dataset (2000–2020).
    IMPORTANT: Annual_Rainfall, Fertilizer, and Pesticide are 100% missing.
    This dataset CANNOT be used for model training with the existing feature schema.
    Use it for district-level exploratory analysis only.
    Returns None if the file is absent.
    """
    if not DATA_PATH_DISTRICT.exists():
        logger.warning("District dataset not found at %s", DATA_PATH_DISTRICT)
        return None
    # Read with low_memory=False due to large file (123 MB)
    df = pd.read_csv(DATA_PATH_DISTRICT, low_memory=False)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].str.strip()
    logger.info("Loaded district dataset: %d rows, %d columns", len(df), df.shape[1])
    return df


def load_official_2021_25_data() -> pd.DataFrame | None:
    """
    Load the official all-India 2020-21 to 2024-25 dataset (225 rows).
    IMPORTANT:
    - Yield column is 100% missing. Supervised validation is NOT possible.
    - Area column is 100% missing. Yield cannot be derived from Production/Area.
    - Geography is All-India only (not state-level).
    - Production unit varies: 'lakh tonnes' for most crops; 'lakh bales' for cotton.
    - Records with Crop_Year=2025 refer to agricultural year 2024-25.
    Use for official production statistics display only.
    Returns None if the file is absent.
    """
    if not DATA_PATH_OFFICIAL_2021_25.exists():
        logger.warning("Official 2021-25 dataset not found at %s", DATA_PATH_OFFICIAL_2021_25)
        return None
    df = pd.read_csv(DATA_PATH_OFFICIAL_2021_25, low_memory=False)
    logger.info("Loaded official 2021-25 dataset: %d rows, %d columns", len(df), df.shape[1])
    return df


def load_combined_data() -> pd.DataFrame | None:
    """
    Load the combined district + all-India dataset (2000–2025, 311,713 rows).
    IMPORTANT:
    - Mix of geography_level='district' and geography_level='all_india'.
    - Annual_Rainfall, Fertilizer, Pesticide are 100% missing.
    - CANNOT be used for model training.
    - Always filter by geography_level before analysis to avoid mixing scales.
    Returns None if the file is absent.
    """
    if not DATA_PATH_COMBINED.exists():
        logger.warning("Combined dataset not found at %s", DATA_PATH_COMBINED)
        return None
    df = pd.read_csv(DATA_PATH_COMBINED, low_memory=False)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].str.strip()
    logger.info("Loaded combined dataset: %d rows, %d columns", len(df), df.shape[1])
    return df


# ── Dataset metadata registry ─────────────────────────────────────────────────
DATASET_REGISTRY = {
    "original": {
        "label":        "Original Historical (1997–2020)",
        "path":         DATA_PATH,
        "years":        "1997–2020",
        "rows":         19_689,
        "geography":    "State-level",
        "model_ready":  True,
        "has_yield":    True,
        "limitations":  "Random 80/20 split may allow (crop, state) overlap between train and test.",
    },
    "model_ready": {
        "label":        "Model-Ready Curated (2000–2020)",
        "path":         DATA_PATH_MODEL_READY,
        "years":        "2000–2020",
        "rows":         17_983,
        "geography":    "State-level",
        "model_ready":  True,
        "has_yield":    True,
        "limitations":  "Covers 2000–2020 only; excludes 1997–1999 records present in the original dataset.",
    },
    "district": {
        "label":        "District-Level Historical (2000–2020)",
        "path":         DATA_PATH_DISTRICT,
        "years":        "2000–2020",
        "rows":         311_488,
        "geography":    "District-level (706 districts, 37 states)",
        "model_ready":  False,
        "has_yield":    True,
        "limitations":  "Annual_Rainfall, Fertilizer, Pesticide are 100% missing — cannot train existing model.",
    },
    "official_2021_25": {
        "label":        "Official All-India 2020-21 to 2024-25",
        "path":         DATA_PATH_OFFICIAL_2021_25,
        "years":        "2020-21 to 2024-25",
        "rows":         225,
        "geography":    "All-India aggregate",
        "model_ready":  False,
        "has_yield":    False,
        "limitations":  (
            "Yield is 100% missing. Area is 100% missing. "
            "Supervised model validation is not possible. "
            "Production in mixed units (lakh tonnes / lakh bales). "
            "Cannot be compared directly to state-level predictions."
        ),
    },
    "combined": {
        "label":        "Combined Dataset (2000–2025)",
        "path":         DATA_PATH_COMBINED,
        "years":        "2000–2025",
        "rows":         311_713,
        "geography":    "Mixed: district + all-India",
        "model_ready":  False,
        "has_yield":    True,  # district rows only
        "limitations":  (
            "Mixes district and all-India geographic levels. "
            "Annual_Rainfall, Fertilizer, Pesticide 100% missing. "
            "Always filter by geography_level before analysis."
        ),
    },
}

