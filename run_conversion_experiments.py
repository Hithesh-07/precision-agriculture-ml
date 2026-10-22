"""
run_conversion_experiments.py
Controlled Coconut Conversion & Model Comparison Experiments:
1. Scenario Baseline: Native coconut units (nuts/ha for coconut, t/ha for other crops).
2. Scenario 1.3 kg: Coconut converted using 1.3 kg/nut (whole fresh nut).
3. Scenario 1.4 kg: Coconut converted using 1.4 kg/nut (whole fresh nut, standard FAO/CDB).
4. Scenario 1.5 kg: Coconut converted using 1.5 kg/nut (heavy whole nut).
5. Approach C (Crop-Specific):
   - Model C1: Dedicated Coconut Model trained and evaluated exclusively in native units (nuts/ha).
   - Model C2: Non-Coconut Model trained and evaluated exclusively on 54 metric-tonne crops (t/ha).
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

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

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    medae = np.median(np.abs(y_true - y_pred))
    p90 = np.percentile(np.abs(y_true - y_pred), 90)
    p95 = np.percentile(np.abs(y_true - y_pred), 95)
    p99 = np.percentile(np.abs(y_true - y_pred), 99)
    neg_count = int(np.sum(y_pred < 0))
    nan_count = int(np.sum(np.isnan(y_pred)))
    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
        "MedAE": round(float(medae), 4),
        "P90": round(float(p90), 4),
        "P95": round(float(p95), 4),
        "P99": round(float(p99), 4),
        "neg_preds": neg_count,
        "nan_preds": nan_count,
    }

def convert_dataset(df: pd.DataFrame, weight_kg: float | None = None) -> pd.DataFrame:
    """
    Safely create converted target without overwriting or double converting.
    If weight_kg is None, returns dataframe with original Yield.
    If weight_kg is provided, creates 'Yield_converted' = Yield * weight_kg / 1000 for Coconut.
    Non-coconut records remain completely untouched.
    """
    d = df.copy()
    if weight_kg is None:
        d["target_eval"] = d[TARGET].copy()
        d["target_unit"] = np.where(d["Crop"].str.strip() == "Coconut", "nuts/ha", "tonnes/ha")
        return d
    
    # Validation check: weight must be positive
    if weight_kg <= 0:
        raise ValueError("Conversion weight must be strictly positive.")
        
    is_coconut = d["Crop"].str.strip() == "Coconut"
    d["target_eval"] = d[TARGET].copy()
    d.loc[is_coconut, "target_eval"] = d.loc[is_coconut, TARGET] * weight_kg / 1000.0
    d["target_unit"] = "tonnes/ha"
    d["conversion_weight_kg"] = np.where(is_coconut, weight_kg, np.nan)
    return d

def run_experiment_on_scenario(df_scenario, scenario_name, target_col="target_eval"):
    print(f"\n=======================================================")
    print(f"EVALUATING SCENARIO: {scenario_name}")
    print(f"=======================================================")
    
    train_idx, test_idx = train_test_split(df_scenario.index, test_size=0.20, random_state=42)
    train_df = df_scenario.loc[train_idx].reset_index(drop=True)
    test_df = df_scenario.loc[test_idx].reset_index(drop=True)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        "LightGBM (log1p)": LGBMRegressor(n_estimators=300, learning_rate=0.045, num_leaves=255, min_child_samples=5, random_state=42, verbose=-1, n_jobs=-1),
        "XGBoost (log1p)": XGBRegressor(n_estimators=250, learning_rate=0.045, max_depth=14, random_state=42, n_jobs=-1),
        "XGBoost (Raw MSE)": XGBRegressor(n_estimators=350, learning_rate=0.04, max_depth=8, min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1),
    }
    
    scenario_res = {}
    
    for mname, est in models.items():
        is_log = "log1p" in mname
        cv_maes, cv_rmses, cv_r2s = [], [], []
        
        for fold, (tr_i, va_i) in enumerate(kf.split(train_df)):
            f_tr = train_df.iloc[tr_i]
            f_va = train_df.iloc[va_i]
            
            X_tr, y_tr_raw = f_tr[ALL_FEATURES], f_tr[target_col].values
            X_va, y_va_raw = f_va[ALL_FEATURES], f_va[target_col].values
            
            y_tr = np.log1p(y_tr_raw) if is_log else y_tr_raw
            
            pipe = Pipeline([("prep", build_preprocessor()), ("model", est)])
            pipe.fit(X_tr, y_tr)
            
            pred = pipe.predict(X_va)
            if is_log:
                pred = np.expm1(np.clip(pred, -1, 20))
            pred = np.clip(pred, 0, None)
            
            m = evaluate(y_va_raw, pred)
            cv_maes.append(m["MAE"])
            cv_rmses.append(m["RMSE"])
            cv_r2s.append(m["R2"])
            
        # Fit on full train and test on untouched test
        X_train_full = train_df[ALL_FEATURES]
        y_train_full = train_df[target_col].values
        X_test_full = test_df[ALL_FEATURES]
        y_test_full = test_df[target_col].values
        
        y_tr_f = np.log1p(y_train_full) if is_log else y_train_full
        full_pipe = Pipeline([("prep", build_preprocessor()), ("model", est)])
        full_pipe.fit(X_train_full, y_tr_f)
        
        test_pred = full_pipe.predict(X_test_full)
        raw_test_pred = test_pred.copy()
        if is_log:
            test_pred = np.expm1(np.clip(test_pred, -1, 20))
        test_pred = np.clip(test_pred, 0, None)
        
        test_m = evaluate(y_test_full, test_pred)
        
        # Chronological Split (Train <= 2015, Test > 2015)
        t_tr = df_scenario[df_scenario["Crop_Year"] <= 2015]
        t_te = df_scenario[df_scenario["Crop_Year"] > 2015]
        y_t_tr = np.log1p(t_tr[target_col].values) if is_log else t_tr[target_col].values
        time_pipe = Pipeline([("prep", build_preprocessor()), ("model", est)])
        time_pipe.fit(t_tr[ALL_FEATURES], y_t_tr)
        time_pred = time_pipe.predict(t_te[ALL_FEATURES])
        if is_log:
            time_pred = np.expm1(np.clip(time_pred, -1, 20))
        time_pred = np.clip(time_pred, 0, None)
        time_m = evaluate(t_te[target_col].values, time_pred)
        
        print(f"\nModel: {mname}")
        print(f"  5-Fold CV: MAE={np.mean(cv_maes):.4f} | RMSE={np.mean(cv_rmses):.4f} | R2={np.mean(cv_r2s):.4f}")
        print(f"  Held-out : MAE={test_m['MAE']:.4f} | RMSE={test_m['RMSE']:.4f} | R2={test_m['R2']:.4f} | MedAE={test_m['MedAE']:.4f}")
        print(f"  Time-CV  : MAE={time_m['MAE']:.4f} | RMSE={time_m['RMSE']:.4f} | R2={time_m['R2']:.4f}")
        
        scenario_res[mname] = {
            "cv": {"MAE": round(float(np.mean(cv_maes)), 4), "RMSE": round(float(np.mean(cv_rmses)), 4), "R2": round(float(np.mean(cv_r2s)), 4)},
            "test": test_m,
            "time": time_m,
        }
    return scenario_res

def main():
    raw_df = clean_data(load_model_ready_data())
    results_all = {}
    
    # 1. Approach A: Baseline native units
    df_base = convert_dataset(raw_df, weight_kg=None)
    results_all["Approach A: Baseline (Native Units)"] = run_experiment_on_scenario(df_base, "Baseline (Native Units)")
    
    # 2. Approach B: Converted weights (1.3, 1.4, 1.5 kg)
    for w in [1.3, 1.4, 1.5]:
        df_conv = convert_dataset(raw_df, weight_kg=w)
        results_all[f"Approach B: Converted ({w} kg/nut)"] = run_experiment_on_scenario(df_conv, f"Converted ({w} kg/nut)")
        
    # 3. Approach C: Crop-Specific Models
    # C1: Non-Coconut Only (54 crops, strictly t/ha)
    print("\n=======================================================")
    print("APPROACH C1: NON-COCONUT CROPS (N=17,826 rows, 54 crops, strictly tonnes/ha)")
    print("=======================================================")
    df_non_coco = raw_df[raw_df["Crop"].str.strip() != "Coconut"].copy().reset_index(drop=True)
    df_non_coco["target_eval"] = df_non_coco[TARGET]
    results_all["Approach C1: Non-Coconut Only (t/ha)"] = run_experiment_on_scenario(df_non_coco, "Non-Coconut Crops (54 Crops, t/ha)")
    
    # C2: Coconut Only (1 crop, 157 rows, native nuts/ha)
    print("\n=======================================================")
    print("APPROACH C2: COCONUT ONLY (N=157 rows, native nuts/ha)")
    print("=======================================================")
    df_coco = raw_df[raw_df["Crop"].str.strip() == "Coconut"].copy().reset_index(drop=True)
    df_coco["target_eval"] = df_coco[TARGET]
    results_all["Approach C2: Coconut Only (nuts/ha)"] = run_experiment_on_scenario(df_coco, "Coconut Only (nuts/ha)")
    
    with open("reports/coconut_conversion_experiments.json", "w") as f:
        json.dump(results_all, f, indent=2)
    print("\nAll experimental results saved to reports/coconut_conversion_experiments.json")

if __name__ == "__main__":
    main()
