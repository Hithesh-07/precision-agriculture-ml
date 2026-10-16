"""
src/model_training.py
=====================
Model training, evaluation, and persistence utilities.
All models are trained with the same preprocessing pipeline and evaluated
on the same held-out test set.

Target transformation:
  Yield is highly right-skewed (skewness ~12.8). We apply log1p transformation
  to stabilize variance and improve linear model performance.
  Predictions are inverse-transformed (expm1) before reporting.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.data_processing import build_preprocessor

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

RANDOM_SEED = 42
TEST_SIZE = 0.2

# ── model definitions ─────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        random_state=RANDOM_SEED,
        max_depth=15,
        min_samples_leaf=5,
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_SEED,
        max_depth=20,
        min_samples_leaf=5,
        n_jobs=-1,
    ),
}

MODEL_DESCRIPTIONS = {
    "Linear Regression": (
        "Fits a linear equation to the features. Fast to train and interpret, "
        "but struggles with non-linear relationships common in agricultural data."
    ),
    "Decision Tree": (
        "Learns a tree of decision rules. Can capture non-linear patterns but "
        "may overfit without depth limits. Single tree, easy to visualise."
    ),
    "Random Forest": (
        "Ensemble of many decision trees. Generally more robust than a single "
        "tree. Provides native feature importance. Slower to train."
    ),
}


# ── split ─────────────────────────────────────────────────────────────────────
def make_split(df: pd.DataFrame, feature_cols: list[str], target_col: str):
    """
    Perform a random 80/20 stratified split on Crop_Year bins.
    Random split is used because the dataset spans 24 years (1997–2020) and
    the same crops / states appear across all years; a purely time-based split
    would make the test set geographically representative but would exclude
    recent years entirely from training. We document this limitation.

    Limitation: rows from the same (crop, state) pair can appear in both sets.
    Future work could use GroupShuffleSplit on (Crop, State) groups.
    """
    X = df[feature_cols]
    y = np.log1p(df[target_col].values)  # log-transform target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    logger.info(
        "Split: %d train / %d test (random, seed=%d)",
        len(X_train), len(X_test), RANDOM_SEED,
    )
    return X_train, X_test, y_train, y_test


# ── metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """
    Compute MAE, RMSE and R² in the ORIGINAL scale (inverse log-transform).
    R² is also reported on the log scale for reference.
    """
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(np.clip(y_pred_log, -1, 20))  # guard against extreme values

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    return {
        "MAE":  round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2":   round(float(r2), 4),
        # also keep log-scale for residual plots
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


# ── train all models ──────────────────────────────────────────────────────────
def train_all_models(X_train, X_test, y_train, y_test, feature_cols: list[str]) -> dict:
    """
    Train all three models and return results dict with metrics and pipelines.
    """
    preprocessor = build_preprocessor()
    results = {}

    for name, estimator in MODEL_CONFIGS.items():
        logger.info("Training: %s ...", name)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)
            results[name] = {
                "pipeline": pipeline,
                "metrics": {k: v for k, v in metrics.items() if k in ("MAE", "RMSE", "R2")},
                "y_true": metrics["y_true"],
                "y_pred": metrics["y_pred"],
            }
            logger.info(
                "%s → MAE=%.4f  RMSE=%.4f  R²=%.4f",
                name, metrics["MAE"], metrics["RMSE"], metrics["R2"],
            )
        except Exception as exc:
            logger.error("Failed to train %s: %s", name, exc)
            results[name] = {"pipeline": None, "error": str(exc)}

    return results


# ── model selection ───────────────────────────────────────────────────────────
def select_best_model(results: dict) -> str:
    """
    Select the best model based on R² on the held-out test set.
    Falls back to RMSE if R² scores are equal.
    Models that failed training are excluded.
    """
    valid = {
        name: res for name, res in results.items()
        if res.get("pipeline") is not None and "metrics" in res
    }
    if not valid:
        raise RuntimeError("No models trained successfully.")

    best = max(valid, key=lambda n: (valid[n]["metrics"]["R2"], -valid[n]["metrics"]["RMSE"]))
    logger.info("Best model selected: %s (R²=%.4f)", best, valid[best]["metrics"]["R2"])
    return best


# ── feature importance ────────────────────────────────────────────────────────
def get_feature_importance(pipeline: Pipeline, feature_names: list[str]) -> dict | None:
    """
    Extract feature importance from the trained pipeline.
    - RandomForest / DecisionTree: use native feature_importances_.
    - LinearRegression: use absolute coefficient values (normalised).
    Returns dict mapping feature_name -> importance (0..1) or None on failure.
    """
    try:
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]

        # Get transformed feature names
        cat_names = feature_names[:3]   # Crop, Season, State
        num_names = feature_names[3:]   # Crop_Year, Area, …

        all_names = cat_names + num_names

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
            importances = importances / importances.sum()
        else:
            return None

        if len(importances) != len(all_names):
            return None

        return dict(zip(all_names, importances.tolist()))

    except Exception as exc:
        logger.warning("Feature importance extraction failed: %s", exc)
        return None


# ── save / load ───────────────────────────────────────────────────────────────
def save_artifacts(best_name: str, results: dict, X_train, X_test) -> None:
    """Save the selected pipeline, all metrics, and evaluation metadata."""
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    best_pipeline = results[best_name]["pipeline"]

    # Save best model pipeline
    joblib.dump(best_pipeline, MODELS_DIR / "best_model.pkl")
    logger.info("Best model saved: models/best_model.pkl")

    # Save all metrics for comparison table
    metrics_all = {}
    for name, res in results.items():
        if res.get("pipeline") is not None and "metrics" in res:
            metrics_all[name] = res["metrics"]

    eval_meta = {
        "best_model": best_name,
        "model_descriptions": MODEL_DESCRIPTIONS,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "random_seed": RANDOM_SEED,
        "test_size_fraction": TEST_SIZE,
        "split_method": "Random (sklearn train_test_split, seed=42)",
        "split_limitation": (
            "Rows from the same (Crop, State) pair may appear in both "
            "train and test sets. Performance may look stronger than on "
            "fully unseen crops or states."
        ),
        "target": "Yield (log1p transformed during training; expm1 for output)",
        "target_unit": "tonnes per hectare (approximately)",
        "trained_at": datetime.now().isoformat(),
        "metrics": metrics_all,
        "y_true_test": results[best_name]["y_true"],
        "y_pred_test": results[best_name]["y_pred"],
    }

    with open(OUTPUTS_DIR / "evaluation.json", "w") as f:
        json.dump(eval_meta, f, indent=2)
    logger.info("Evaluation metadata saved: outputs/evaluation.json")


def load_best_model() -> Pipeline:
    """Load the saved best-model pipeline. Raises FileNotFoundError if missing."""
    model_path = MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            "No trained model found at models/best_model.pkl. "
            "Run train_model.py to train the models."
        )
    pipeline = joblib.load(model_path)
    return pipeline


def load_evaluation() -> dict:
    """Load evaluation results. Raises FileNotFoundError if missing."""
    eval_path = OUTPUTS_DIR / "evaluation.json"
    if not eval_path.exists():
        raise FileNotFoundError(
            "No evaluation results found at outputs/evaluation.json. "
            "Run train_model.py first."
        )
    with open(eval_path) as f:
        return json.load(f)
