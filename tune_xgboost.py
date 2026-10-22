"""
tune_xgboost.py
Phase 5: Fine-Grained Hyperparameter Tuning for XGBoost
Tuning max_depth, learning_rate, n_estimators, min_child_weight, colsample_bytree, subsample, reg_alpha, reg_lambda
Evaluating via 5-Fold Cross-Validation on N=14,386 training split
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

sys.path.insert(0, str(Path(__file__).parent))
from src.data_processing import TARGET, clean_data, load_model_ready_data

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["fertilizer_per_ha"] = d["Fertilizer"] / d["Area"]
    d["pesticide_per_ha"] = d["Pesticide"] / d["Area"]
    d["Crop_Season"] = d["Crop"].astype(str) + "_" + d["Season"].astype(str)
    d["rainfall_per_area"] = d["Annual_Rainfall"] / (d["Area"] + 1.0)
    return d

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def main():
    raw_df = clean_data(load_model_ready_data())
    df_feat = build_features(raw_df)
    
    eng_cats = ["Crop", "Season", "State", "Crop_Season"]
    eng_nums = ["Crop_Year", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide", "fertilizer_per_ha", "pesticide_per_ha", "rainfall_per_area"]
    eng_features = eng_cats + eng_nums
    
    train_idx, test_idx = train_test_split(df_feat.index, test_size=0.20, random_state=42)
    train_df = df_feat.loc[train_idx].reset_index(drop=True)
    test_df = df_feat.loc[test_idx].reset_index(drop=True)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), eng_cats),
        ("num", StandardScaler(), eng_nums),
    ])
    
    param_grid = [
        # Candidate 1: deeper trees, moderate learning rate
        {"max_depth": 10, "learning_rate": 0.04, "n_estimators": 400, "min_child_weight": 2, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.1, "reg_lambda": 1.0},
        # Candidate 2: deeper trees (depth 12), lower learning rate
        {"max_depth": 12, "learning_rate": 0.03, "n_estimators": 450, "min_child_weight": 3, "subsample": 0.85, "colsample_bytree": 0.85, "reg_alpha": 0.05, "reg_lambda": 0.5},
        # Candidate 3: depth 14, lr 0.035
        {"max_depth": 14, "learning_rate": 0.035, "n_estimators": 400, "min_child_weight": 2, "subsample": 0.90, "colsample_bytree": 0.85, "reg_alpha": 0.05, "reg_lambda": 0.5},
        # Candidate 4: depth 8, more estimators (fast, regularized)
        {"max_depth": 8, "learning_rate": 0.05, "n_estimators": 500, "min_child_weight": 1, "subsample": 0.80, "colsample_bytree": 0.80, "reg_alpha": 0.2, "reg_lambda": 1.5},
    ]
    
    best_candidate = None
    best_cv_rmse = float("inf")
    results = []
    
    print("\n--- TUNING XGBOOST WITH 5-FOLD CV ---")
    for i, p in enumerate(param_grid):
        model = XGBRegressor(**p, random_state=42, n_jobs=-1)
        cv_maes, cv_rmses, cv_r2s = [], [], []
        
        for fold, (tr_i, va_i) in enumerate(kf.split(train_df)):
            f_tr = train_df.iloc[tr_i]
            f_va = train_df.iloc[va_i]
            
            X_tr, y_tr = f_tr[eng_features], f_tr[TARGET].values
            X_va, y_va = f_va[eng_features], f_va[TARGET].values
            
            pipe = Pipeline([("prep", preprocessor), ("model", model)])
            pipe.fit(X_tr, y_tr)
            pred = np.clip(pipe.predict(X_va), 0, None)
            
            m = evaluate(y_va, pred)
            cv_maes.append(m["MAE"])
            cv_rmses.append(m["RMSE"])
            cv_r2s.append(m["R2"])
            
        mean_mae = np.mean(cv_maes)
        mean_rmse = np.mean(cv_rmses)
        mean_r2 = np.mean(cv_r2s)
        
        print(f"Candidate {i+1}: depth={p['max_depth']}, lr={p['learning_rate']}, n_est={p['n_estimators']}, min_child={p['min_child_weight']}")
        print(f"  CV MAE : {mean_mae:.4f} (+/- {np.std(cv_maes):.4f})")
        print(f"  CV RMSE: {mean_rmse:.4f} (+/- {np.std(cv_rmses):.4f})")
        print(f"  CV R2  : {mean_r2:.4f} (+/- {np.std(cv_r2s):.4f})")
        
        # Test on held-out test split
        full_pipe = Pipeline([("prep", preprocessor), ("model", model)])
        full_pipe.fit(train_df[eng_features], train_df[TARGET].values)
        test_pred = np.clip(full_pipe.predict(test_df[eng_features]), 0, None)
        test_m = evaluate(test_df[TARGET].values, test_pred)
        print(f"  --> HELD-OUT TEST: MAE={test_m['MAE']:.4f} | RMSE={test_m['RMSE']:.4f} | R2={test_m['R2']:.4f}\n")
        
        results.append({
            "params": p,
            "cv_mae": mean_mae,
            "cv_rmse": mean_rmse,
            "cv_r2": mean_r2,
            "test_mae": test_m["MAE"],
            "test_rmse": test_m["RMSE"],
            "test_r2": test_m["R2"],
        })
        
        if mean_rmse < best_cv_rmse:
            best_cv_rmse = mean_rmse
            best_candidate = results[-1]
            
    with open("reports/xgboost_tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Best candidate saved to reports/xgboost_tuning_results.json")

if __name__ == "__main__":
    main()
