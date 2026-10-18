"""
train_model.py
==============
Training script for the CropZen ML pipeline.

Usage:
    .venv\\Scripts\\python.exe train_model.py
    .venv\\Scripts\\python.exe train_model.py --dataset model_ready
    .venv\\Scripts\\python.exe train_model.py --dataset original
    .venv\\Scripts\\python.exe train_model.py --split time  # time-based split: train ≤2015, test >2015

Supported datasets for training:
  original    — data/crop_yield.csv (1997–2020, 19,689 rows)  [default]
  model_ready — data/cropzen_model_ready_2000_2020.csv (2000–2020, 17,983 rows)

The district, official_2021_25, and combined datasets cannot be used for
training because they are missing required features (Annual_Rainfall,
Fertilizer, Pesticide) or have no observed Yield values.

This script:
1. Loads and validates the raw dataset.
2. Cleans data (documents all decisions).
3. Splits data into train and test sets (random 80/20 or time-based).
4. Trains Linear Regression, Decision Tree, and Random Forest.
5. Evaluates all models on the same held-out test set.
6. Saves the best model pipeline and evaluation artifacts.

The script must be run at least once before the Streamlit app can
make predictions. Never hardcode sample metrics — results are always
computed from actual training.
"""

import argparse
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
    DATASET_REGISTRY,
    clean_data,
    load_raw_data,
    load_model_ready_data,
    save_schema_info,
    validate_schema,
)
from src.model_training import (
    make_split,
    make_time_split,
    save_artifacts,
    select_best_model,
    train_all_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CropZen yield prediction models."
    )
    parser.add_argument(
        "--dataset",
        choices=["original", "model_ready"],
        default="model_ready",
        help=(
            "Which dataset to train on. "
            "'model_ready' = cropzen_model_ready_2000_2020.csv (2000-2020, 17,983 rows, default). "
            "'original' = data/crop_yield.csv (1997-2020, 19,689 rows)."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["random", "time"],
        default="random",
        help=(
            "Split strategy. "
            "'random' = 80/20 random split (seed=42, default). "
            "'time' = time-based: train on years <= SPLIT_YEAR, test on later years."
        ),
    )
    parser.add_argument(
        "--split-year",
        type=int,
        default=2015,
        help="Year cutoff for time-based split (inclusive train cutoff). Default: 2015.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the full training pipeline. Returns 0 on success, 1 on failure."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("CropZen — Model Training")
    logger.info("=" * 60)
    logger.info("Dataset  : %s", args.dataset)
    logger.info("Split    : %s", args.split)
    if args.split == "time":
        logger.info("Split year: train <= %d, test > %d", args.split_year, args.split_year)

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    logger.info("Step 1: Loading dataset...")
    try:
        if args.dataset == "model_ready":
            df = load_model_ready_data()
            if df is None:
                logger.error(
                    "Model-ready dataset not found. "
                    "Copy cropzen_model_ready_2000_2020.csv into the data/ directory."
                )
                return 1
            dataset_label = DATASET_REGISTRY["model_ready"]["label"]
            dataset_file  = "cropzen_model_ready_2000_2020.csv"
        else:
            df = load_raw_data()
            dataset_label = DATASET_REGISTRY["original"]["label"]
            dataset_file  = "crop_yield.csv"
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1

    logger.info("Loaded %d rows, %d columns from %s.", df.shape[0], df.shape[1], dataset_file)

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
    if args.split == "time":
        logger.info(
            "Step 5: Time-based split — train: Crop_Year <= %d, test: Crop_Year > %d",
            args.split_year, args.split_year,
        )
        logger.info(
            "NOTE: Time-based split tests generalization to future years. "
            "Train and test sets are mutually exclusive by year. "
            "Metrics will typically be lower than random-split because the model "
            "is never trained on any record from the test years."
        )
        split_result = make_time_split(df, ALL_FEATURES, TARGET, args.split_year)
        if split_result is None:
            logger.error("Time-based split failed (possibly no data after split year %d).", args.split_year)
            return 1
        X_train, X_test, y_train, y_test = split_result
        split_method = f"Time-based (train ≤ {args.split_year}, test > {args.split_year})"
        split_limitation = (
            "Time-based split: the model is tested on years it was never trained on, "
            "providing a more realistic assessment of out-of-sample performance."
        )
    else:
        logger.info("Step 5: Random 80/20 split (seed=42)...")
        logger.info(
            "NOTE: Random split used. Rows from the same (Crop, State) pair may appear "
            "in both train and test sets, potentially inflating test performance."
        )
        X_train, X_test, y_train, y_test = make_split(df, ALL_FEATURES, TARGET)
        split_method = "Random 80/20 (sklearn train_test_split, seed=42)"
        split_limitation = (
            "Rows from the same (Crop, State) pair may appear in both train and test sets. "
            "Performance may appear stronger than on fully unseen crops or states."
        )

    logger.info("Train: %d rows  |  Test: %d rows", len(X_train), len(X_test))

    # ── Step 6: Train all models ──────────────────────────────────────────────
    logger.info("Step 6: Training all models...")
    logger.info("Target: log1p(Yield) during training; expm1 applied to predictions.")

    results = train_all_models(X_train, X_test, y_train, y_test, ALL_FEATURES)

    # ── Step 7: Report metrics ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION RESULTS (original scale, t/ha)")
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
    logger.info("Step 7: Selecting best model (criterion: highest R² on test set)...")
    best_name = select_best_model(results)
    logger.info("Best model: %s", best_name)

    # ── Step 9: Save artifacts ────────────────────────────────────────────────
    logger.info("Step 8: Saving model pipeline and evaluation artifacts...")
    save_artifacts(
        best_name, results, X_train, X_test,
        extra_meta={
            "dataset_label":      dataset_label,
            "dataset_file":       dataset_file,
            "split_method":       split_method,
            "split_limitation":   split_limitation,
            "split_year_cutoff":  args.split_year if args.split == "time" else None,
        },
    )

    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info("Run the app: .venv\\Scripts\\python.exe -m streamlit run app.py")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
