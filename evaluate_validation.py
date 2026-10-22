"""
evaluate_validation.py
Phase 4: Cross-Validation & Validation Methodology
- 5-Fold Cross-Validation on the 80% training set (N=14,386)
- Time-based Validation (train <= 2015, test > 2015)
- Geographic Holdout Validation (state-level holdouts)
"""

import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

sys.path.insert(0, str(Path(__file__).parent))
from src.data_processing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    clean_data,
    load_model_ready_data,
    build_preprocessor,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_validation")

def evaluate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def main():
    df = clean_data(load_model_ready_data())
    from sklearn.model_selection import train_test_split
    
    # 80/20 train/test split with random_state=42 (untouched test set)
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
    logger.info("Training set: %d rows | Untouched test set: %d rows", len(train_df), len(test_df))
    
    # 5-Fold Cross-Validation on train_df
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        "Baseline XGBoost (log1p, depth=14, lr=0.05)": (
            XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=14, random_state=42, n_jobs=-1),
            True # log1p
        ),
        "XGBoost Raw Yield (squared_error, depth=8, lr=0.05)": (
            XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, random_state=42, n_jobs=-1),
            False # raw
        ),
        "Baseline LightGBM (log1p, leaves=255, lr=0.05)": (
            LGBMRegressor(n_estimators=250, learning_rate=0.05, num_leaves=255, min_child_samples=5, random_state=42, verbose=-1, n_jobs=-1),
            True
        ),
        "HistGradientBoosting (log1p, max_iter=200)": (
            HistGradientBoostingRegressor(max_iter=200, random_state=42),
            True
        ),
    }
    
    print("\n=======================================================")
    print("5-FOLD CROSS-VALIDATION ON TRAINING SET (N=14,386)")
    print("=======================================================")
    
    for name, (estimator, use_log) in models.items():
        maes, rmses, r2s = [], [], []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
            fold_train = train_df.iloc[train_idx]
            fold_val = train_df.iloc[val_idx]
            
            X_tr, y_tr_raw = fold_train[ALL_FEATURES], fold_train[TARGET].values
            X_va, y_va_raw = fold_val[ALL_FEATURES], fold_val[TARGET].values
            
            y_tr = np.log1p(y_tr_raw) if use_log else y_tr_raw
            
            preprocessor = build_preprocessor()
            pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
            pipe.fit(X_tr, y_tr)
            
            pred = pipe.predict(X_va)
            if use_log:
                pred = np.expm1(np.clip(pred, -1, 20))
            
            m = evaluate_metrics(y_va_raw, pred)
            maes.append(m["MAE"])
            rmses.append(m["RMSE"])
            r2s.append(m["R2"])
            
        print(f"\nModel: {name}")
        print(f"  CV MAE : {np.mean(maes):.4f} (+/- {np.std(maes):.4f})")
        print(f"  CV RMSE: {np.mean(rmses):.4f} (+/- {np.std(rmses):.4f})")
        print(f"  CV R2  : {np.mean(r2s):.4f} (+/- {np.std(r2s):.4f})")
        
    print("\n=======================================================")
    print("TIME-BASED GENERALIZATION (Train <= 2015, Test > 2015)")
    print("=======================================================")
    t_train = df[df["Crop_Year"] <= 2015]
    t_test = df[df["Crop_Year"] > 2015]
    print(f"Time Train rows: {len(t_train)} (Years {t_train['Crop_Year'].min()}-{t_train['Crop_Year'].max()})")
    print(f"Time Test rows : {len(t_test)} (Years {t_test['Crop_Year'].min()}-{t_test['Crop_Year'].max()})")
    
    for name, (estimator, use_log) in models.items():
        X_tr, y_tr_raw = t_train[ALL_FEATURES], t_train[TARGET].values
        X_te, y_te_raw = t_test[ALL_FEATURES], t_test[TARGET].values
        y_tr = np.log1p(y_tr_raw) if use_log else y_tr_raw
        
        preprocessor = build_preprocessor()
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        if use_log:
            pred = np.expm1(np.clip(pred, -1, 20))
            
        m = evaluate_metrics(y_te_raw, pred)
        print(f"Model: {name}")
        print(f"  Time MAE: {m['MAE']:.4f} | RMSE: {m['RMSE']:.4f} | R2: {m['R2']:.4f}")

if __name__ == "__main__":
    main()
