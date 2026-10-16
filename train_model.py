"""
train_model.py
==============
Standalone training script for the Precision Agriculture ML pipeline.

Usage:
    .venv\\Scripts\\python.exe train_model.py

This script:
1. Loads and validates the raw dataset.
2. Cleans data (documents all decisions).
3. Splits data into train and test sets.
4. Trains Linear Regression, Decision Tree, and Random Forest.
5. Evaluates all models on the same held-out test set.
6. Saves the best model pipeline and evaluation artifacts.

The script must be run at least once before the Streamlit app can
make predictions. Never hardcode sample metrics — results are always
computed from actual training.
"""

import logging
import sys
from pathlib import Path

# Add project root to path so src imports work
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_model")

from src.data_processing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    clean_data,
    load_raw_data,
    save_schema_info,
    validate_schema,
)
from src.model_training import (
    make_split,
    save_artifacts,
    select_best_model,
    train_all_models,
)


def main() -> int:
    """Run the full training pipeline. Returns 0 on success, 1 on failure."""

    logger.info("=" * 60)
    logger.info("Precision Agriculture — Model Training")
    logger.info("=" * 60)

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    logger.info("Step 1: Loading dataset...")
    try:
        df = load_raw_data()
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1

    logger.info("Loaded %d rows, %d columns.", df.shape[0], df.shape[1])

    # ── Step 2: Validate schema ───────────────────────────────────────────────
    logger.info("Step 2: Validating schema...")
    try:
        validate_schema(df)
        logger.info("Schema OK.")
    except ValueError as e:
        logger.error("Schema validation failed: %s", e)
        return 1

    # ── Step 3: Clean data ────────────────────────────────────────────────────
    logger.info("Step 3: Cleaning data...")
    df = clean_data(df, log_decisions=True)
    logger.info("After cleaning: %d rows remain.", len(df))

    if len(df) < 100:
        logger.error("Too few rows after cleaning (%d). Aborting.", len(df))
        return 1

    # ── Step 4: Save schema info ──────────────────────────────────────────────
    logger.info("Step 4: Saving feature schema from training data...")
    save_schema_info(df)

    # ── Step 5: Split ─────────────────────────────────────────────────────────
    logger.info("Step 5: Splitting into train / test sets (80/20, seed=42)...")
    logger.info("Note: Random split used. See src/model_training.py for limitations.")
    X_train, X_test, y_train, y_test = make_split(df, ALL_FEATURES, TARGET)
    logger.info("Train: %d rows  |  Test: %d rows", len(X_train), len(X_test))

    # ── Step 6: Train all models ──────────────────────────────────────────────
    logger.info("Step 6: Training all models...")
    logger.info(
        "Target: log1p(Yield) during training; expm1 applied to predictions."
    )

    results = train_all_models(X_train, X_test, y_train, y_test, ALL_FEATURES)

    # ── Step 7: Report metrics ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION RESULTS (original scale)")
    logger.info("=" * 60)
    any_success = False
    for name, res in results.items():
        if res.get("pipeline") is None:
            logger.warning("%s: FAILED — %s", name, res.get("error", "unknown"))
        else:
            m = res["metrics"]
            logger.info(
                "%-22s  MAE=%8.4f  RMSE=%9.4f  R²=%7.4f",
                name, m["MAE"], m["RMSE"], m["R2"],
            )
            any_success = True

    if not any_success:
        logger.error("All models failed. No artifacts saved.")
        return 1

    # ── Step 8: Select best model ─────────────────────────────────────────────
    logger.info("Step 7: Selecting best model (criterion: highest R²)...")
    best_name = select_best_model(results)
    logger.info("Best model: %s", best_name)

    # ── Step 9: Save artifacts ────────────────────────────────────────────────
    logger.info("Step 8: Saving model pipeline and evaluation artifacts...")
    save_artifacts(best_name, results, X_train, X_test)

    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info("Run the app: .venv\\Scripts\\python.exe -m streamlit run app.py")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
