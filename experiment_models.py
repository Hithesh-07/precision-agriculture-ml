"""
experiment_models.py
Phases 5, 6, 7, 8:
- Feature Engineering (fertilizer_per_ha, pesticide_per_ha, rainfall_per_area, crop-season interaction)
- Loss functions (squared_error vs Huber / pseudo-huber)
- Target transformations (raw vs log1p vs Box-Cox/PowerTransform)
- Model Architectures (Tuned XGBoost, Tuned LightGBM, CatBoost, HistGradientBoosting, ExtraTrees, RandomForest)
- 5-Fold Cross Validation on N=14,386 training split
- Final honest evaluation on held-out N=3,597 test split
"""

import sys
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)

sys.path.insert(0, str(Path(__file__).parent))
from src.data_processing import (
    TARGET,
    clean_data,
    load_model_ready_data,
)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer domain-valid agricultural intensity features without leakage."""
    d = df.copy()
    # Intensity features (per unit area)
    # Area > 0 is guaranteed by clean_data
    d["fertilizer_per_ha"] = d["Fertilizer"] / d["Area"]
    d["pesticide_per_ha"] = d["Pesticide"] / d["Area"]
    
    # Interaction: Crop + Season
    d["Crop_Season"] = d["Crop"].astype(str) + "_" + d["Season"].astype(str)
    
    # Rainfall intensity per area
    d["rainfall_per_area"] = d["Annual_Rainfall"] / (d["Area"] + 1.0)
    
    return d

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def run_experiments():
    raw_df = clean_data(load_model_ready_data())
    df_feat = build_features(raw_df)
    
    # Define feature sets
    base_cats = ["Crop", "Season", "State"]
    base_nums = ["Crop_Year", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]
    base_features = base_cats + base_nums
    
    eng_cats = ["Crop", "Season", "State", "Crop_Season"]
    eng_nums = ["Crop_Year", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide", "fertilizer_per_ha", "pesticide_per_ha", "rainfall_per_area"]
    eng_features = eng_cats + eng_nums
    
    # 80/20 train/test split on fixed random_state=42
    train_idx, test_idx = train_test_split(df_feat.index, test_size=0.20, random_state=42)
    train_df = df_feat.loc[train_idx].reset_index(drop=True)
    test_df = df_feat.loc[test_idx].reset_index(drop=True)
    
    print(f"Total rows: {len(df_feat)} | Train: {len(train_df)} | Held-out Test: {len(test_df)}")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Experiments definition: (Name, model_fn, features, use_log, use_huber)
    candidate_configs = [
        # Baseline
        {
            "name": "1. Baseline XGBoost (log1p, base features)",
            "estimator": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=14, random_state=42, n_jobs=-1),
            "features": base_features,
            "cats": base_cats,
            "nums": base_nums,
            "target_transform": "log1p",
        },
        # XGBoost on Raw Yield (RMSE directly optimized)
        {
            "name": "2. XGBoost Raw Yield (squared_error, base features)",
            "estimator": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=10, min_child_weight=2, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
            "features": base_features,
            "cats": base_cats,
            "nums": base_nums,
            "target_transform": "raw",
        },
        # XGBoost on Raw Yield with Engineered Features
        {
            "name": "3. XGBoost Raw Yield + Engineered Features",
            "estimator": XGBRegressor(n_estimators=350, learning_rate=0.04, max_depth=10, min_child_weight=2, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1),
            "features": eng_features,
            "cats": eng_cats,
            "nums": eng_nums,
            "target_transform": "raw",
        },
        # XGBoost with Engineered Features (log1p)
        {
            "name": "4. XGBoost log1p + Engineered Features (Tuned)",
            "estimator": XGBRegressor(n_estimators=300, learning_rate=0.04, max_depth=12, min_child_weight=2, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=0.5, random_state=42, n_jobs=-1),
            "features": eng_features,
            "cats": eng_cats,
            "nums": eng_nums,
            "target_transform": "log1p",
        },
        # LightGBM Tuned with Engineered Features (log1p)
        {
            "name": "5. LightGBM log1p + Engineered Features (Tuned)",
            "estimator": LGBMRegressor(n_estimators=350, learning_rate=0.04, num_leaves=127, min_child_samples=10, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=-1),
            "features": eng_features,
            "cats": eng_cats,
            "nums": eng_nums,
            "target_transform": "log1p",
        },
        # LightGBM Raw Yield (huber loss to prevent extreme outliers)
        {
            "name": "6. LightGBM Raw Huber Loss + Engineered Features",
            "estimator": LGBMRegressor(objective="huber", alpha=0.9, n_estimators=350, learning_rate=0.05, num_leaves=127, min_child_samples=10, random_state=42, verbose=-1, n_jobs=-1),
            "features": eng_features,
            "cats": eng_cats,
            "nums": eng_nums,
            "target_transform": "raw",
        },
        # HistGradientBoosting (log1p)
        {
            "name": "7. HistGradientBoosting (log1p + Eng Features)",
            "estimator": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=10, random_state=42),
            "features": eng_features,
            "cats": eng_cats,
            "nums": eng_nums,
            "target_transform": "log1p",
        },
        # Extra Trees Regressor (log1p)
        {
            "name": "8. Extra Trees Regressor (log1p + Eng Features)",
            "estimator": ExtraTreesRegressor(n_estimators=200, max_depth=25, min_samples_leaf=3, random_state=42, n_jobs=-1),
            "features": eng_features,
            "cats": eng_cats,
            "nums": eng_nums,
            "target_transform": "log1p",
        },
        # Random Forest Regressor (log1p)
        {
            "name": "9. Random Forest Regressor (log1p + Eng Features)",
            "estimator": RandomForestRegressor(n_estimators=200, max_depth=25, min_samples_leaf=3, random_state=42, n_jobs=-1),
            "features": eng_features,
            "cats": eng_cats,
            "nums": eng_nums,
            "target_transform": "log1p",
        },
    ]
    
    summary_results = []
    
    print("\n--- RUNNING 5-FOLD CV ON TRAINING SET ---")
    for cfg in candidate_configs:
        start_t = time.time()
        cv_maes, cv_rmses, cv_r2s = [], [], []
        
        preprocessor = ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cfg["cats"]),
            ("num", StandardScaler(), cfg["nums"]),
        ])
        
        for fold, (tr_i, va_i) in enumerate(kf.split(train_df)):
            f_tr = train_df.iloc[tr_i]
            f_va = train_df.iloc[va_i]
            
            X_tr, y_tr_raw = f_tr[cfg["features"]], f_tr[TARGET].values
            X_va, y_va_raw = f_va[cfg["features"]], f_va[TARGET].values
            
            y_tr = np.log1p(y_tr_raw) if cfg["target_transform"] == "log1p" else y_tr_raw
            
            pipe = Pipeline([("prep", preprocessor), ("model", cfg["estimator"])])
            pipe.fit(X_tr, y_tr)
            
            pred = pipe.predict(X_va)
            if cfg["target_transform"] == "log1p":
                pred = np.expm1(np.clip(pred, -1, 20))
            pred = np.clip(pred, 0, None) # Yield cannot be negative
            
            m = evaluate(y_va_raw, pred)
            cv_maes.append(m["MAE"])
            cv_rmses.append(m["RMSE"])
            cv_r2s.append(m["R2"])
            
        elapsed = time.time() - start_t
        mean_mae = np.mean(cv_maes)
        mean_rmse = np.mean(cv_rmses)
        mean_r2 = np.mean(cv_r2s)
        
        print(f"\n{cfg['name']}")
        print(f"  CV MAE : {mean_mae:.4f} (+/- {np.std(cv_maes):.4f})")
        print(f"  CV RMSE: {mean_rmse:.4f} (+/- {np.std(cv_rmses):.4f})")
        print(f"  CV R2  : {mean_r2:.4f} (+/- {np.std(cv_r2s):.4f})")
        print(f"  Time   : {elapsed:.1f}s")
        
        # Now fit on full train_df and evaluate on untouched test_df
        X_train_full = train_df[cfg["features"]]
        y_train_full = train_df[TARGET].values
        X_test_full = test_df[cfg["features"]]
        y_test_full = test_df[TARGET].values
        
        y_tr_f = np.log1p(y_train_full) if cfg["target_transform"] == "log1p" else y_train_full
        
        full_pipe = Pipeline([("prep", preprocessor), ("model", cfg["estimator"])])
        full_pipe.fit(X_train_full, y_tr_f)
        
        test_pred = full_pipe.predict(X_test_full)
        if cfg["target_transform"] == "log1p":
            test_pred = np.expm1(np.clip(test_pred, -1, 20))
        test_pred = np.clip(test_pred, 0, None)
        
        test_m = evaluate(y_test_full, test_pred)
        print(f"  --> HELD-OUT TEST METRICS: MAE={test_m['MAE']:.4f} | RMSE={test_m['RMSE']:.4f} | R2={test_m['R2']:.4f}")
        
        summary_results.append({
            "name": cfg["name"],
            "cv_mae": mean_mae,
            "cv_rmse": mean_rmse,
            "cv_r2": mean_r2,
            "test_mae": test_m["MAE"],
            "test_rmse": test_m["RMSE"],
            "test_r2": test_m["R2"],
            "elapsed_sec": round(elapsed, 1),
        })

    # Save summary table as JSON
    with open("reports/model_comparison_results.json", "w") as f:
        json.dump(summary_results, f, indent=2)
    print("\nSummary written to reports/model_comparison_results.json")

if __name__ == "__main__":
    run_experiments()
