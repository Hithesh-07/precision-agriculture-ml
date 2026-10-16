"""
data_audit.py
=============
Data audit script for the crop_yield.csv dataset.
Inspects schema, data quality, target definition, and potential leakage issues.
Run: .venv\Scripts\python.exe data_audit.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# ── locate dataset ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
POSSIBLE_PATHS = [
    BASE_DIR / "data" / "crop_yield.csv",
    BASE_DIR / "crop_yield.csv",
]

csv_path = None
for p in POSSIBLE_PATHS:
    if p.exists():
        csv_path = p
        break

if csv_path is None:
    print("ERROR: crop_yield.csv not found. Checked:")
    for p in POSSIBLE_PATHS:
        print(f"  {p}")
    sys.exit(1)

print(f"Dataset found: {csv_path}")
df = pd.read_csv(csv_path)

# ── shape ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SHAPE")
print("="*60)
print(f"Rows   : {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

# ── columns and dtypes ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("COLUMNS AND DTYPES")
print("="*60)
for col in df.columns:
    print(f"  {col:<20} {str(df[col].dtype):<12} nunique={df[col].nunique()}")

# ── missing values ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MISSING VALUES")
print("="*60)
missing = df.isnull().sum()
has_missing = False
for col, cnt in missing.items():
    if cnt > 0:
        has_missing = True
        pct = cnt / len(df) * 100
        print(f"  {col:<20} {cnt:>6} missing  ({pct:.2f}%)")
if not has_missing:
    print("  No missing values found.")

# ── duplicates ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DUPLICATE ROWS")
print("="*60)
dup_count = df.duplicated().sum()
print(f"  Exact duplicate rows: {dup_count:,}")

# ── categorical exploration ───────────────────────────────────────────────────
print("\n" + "="*60)
print("CATEGORICAL COLUMNS")
print("="*60)
for col in ["Crop", "Season", "State"]:
    if col in df.columns:
        vals = df[col].str.strip().unique()
        print(f"\n{col}: {len(vals)} unique values")
        for v in sorted(vals)[:15]:
            cnt = (df[col].str.strip() == v).sum()
            print(f"    {v:<35} n={cnt:>5}")
        if len(vals) > 15:
            print(f"    ... and {len(vals)-15} more")

# ── year range ────────────────────────────────────────────────────────────────
if "Crop_Year" in df.columns:
    print("\n" + "="*60)
    print("CROP_YEAR RANGE")
    print("="*60)
    print(f"  Min: {df['Crop_Year'].min()}   Max: {df['Crop_Year'].max()}")
    print(f"  Unique years: {sorted(df['Crop_Year'].unique().tolist())}")

# ── numeric descriptive stats ────────────────────────────────────────────────
numeric_cols = ["Area", "Production", "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"]
numeric_cols = [c for c in numeric_cols if c in df.columns]

print("\n" + "="*60)
print("NUMERIC DESCRIPTIVE STATISTICS")
print("="*60)
print(df[numeric_cols].describe().to_string())

# ── zero / negative values ────────────────────────────────────────────────────
print("\n" + "="*60)
print("ZERO AND NEGATIVE VALUES")
print("="*60)
for col in numeric_cols:
    zeros = (df[col] == 0).sum()
    negs  = (df[col] < 0).sum()
    if zeros > 0 or negs > 0:
        print(f"  {col:<20} zeros={zeros:>5}  negatives={negs:>5}")
    else:
        print(f"  {col:<20} zeros=    0  negatives=    0  OK")

# ── target leakage check ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("TARGET LEAKAGE CHECK  (Yield vs Production/Area)")
print("="*60)
if all(c in df.columns for c in ["Production", "Area", "Yield"]):
    df_c = df.dropna(subset=["Production", "Area", "Yield"])
    df_c = df_c[df_c["Area"] > 0]
    derived = df_c["Production"] / df_c["Area"]
    diff = (derived - df_c["Yield"]).abs()
    max_diff  = diff.max()
    mean_diff = diff.mean()
    exact_pct = (diff < 0.001).sum() / len(df_c) * 100
    print(f"  Max  |Production/Area - Yield|: {max_diff:.8f}")
    print(f"  Mean |Production/Area - Yield|: {mean_diff:.8f}")
    print(f"  Rows where derived == Yield (within 0.001): {exact_pct:.1f}%")
    if exact_pct > 99:
        print("\n  CONCLUSION: Yield IS derived from Production / Area.")
        print("  => 'Production' must NOT be used as a predictor (data leakage).")
        print("  => 'Area' is a valid input because it is known BEFORE harvest.")
    else:
        print("\n  CONCLUSION: Yield and Production/Area do not match exactly.")
        print("  => Investigate further before using Production as a predictor.")

# ── yield distribution ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("YIELD DISTRIBUTION")
print("="*60)
if "Yield" in df.columns:
    y = df["Yield"].dropna()
    q1, q3 = y.quantile(0.25), y.quantile(0.75)
    iqr = q3 - q1
    print(f"  Min        : {y.min():.4f}")
    print(f"  25th pct   : {q1:.4f}")
    print(f"  Median     : {y.median():.4f}")
    print(f"  Mean       : {y.mean():.4f}")
    print(f"  75th pct   : {q3:.4f}")
    print(f"  Max        : {y.max():.4f}")
    print(f"  Std Dev    : {y.std():.4f}")
    print(f"  Skewness   : {y.skew():.4f}")
    print(f"  99th pct   : {y.quantile(0.99):.4f}")
    outlier_high = (y > q3 + 3*iqr).sum()
    outlier_low  = (y < q1 - 3*iqr).sum()
    print(f"  Outliers (3xIQR above): {outlier_high}")
    print(f"  Outliers (3xIQR below): {outlier_low}")
    print(f"  Zero-yield rows       : {(y == 0).sum()}")

# ── correlations ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("CORRELATIONS WITH YIELD")
print("="*60)
safe_cols = [c for c in ["Area", "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"] if c in df.columns]
corr = df[safe_cols].corr()["Yield"].drop("Yield")
for col, val in corr.sort_values(ascending=False).items():
    print(f"  {col:<20} r = {val:+.4f}")
print("  (Correlation does not imply causation)")

# ── final feature recommendation ─────────────────────────────────────────────
print("\n" + "="*60)
print("RECOMMENDED FEATURE SCHEMA (for model training)")
print("="*60)
print("  INPUT FEATURES:")
print("    Categorical : Crop, Season, State")
print("    Numeric     : Crop_Year, Area, Annual_Rainfall, Fertilizer, Pesticide")
print("  TARGET:")
print("    Yield  (tonnes per hectare — derived from Production/Area)")
print("  EXCLUDED:")
print("    Production  — leaks the target (Yield = Production / Area)")
print()
print("Data audit complete.")

# ── save audit summary ────────────────────────────────────────────────────────
os.makedirs(BASE_DIR / "outputs", exist_ok=True)
summary = {
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "missing_total": int(df.isnull().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "unique_crops": int(df["Crop"].nunique()) if "Crop" in df.columns else None,
    "unique_states": int(df["State"].nunique()) if "State" in df.columns else None,
    "unique_seasons": int(df["Season"].nunique()) if "Season" in df.columns else None,
    "year_min": int(df["Crop_Year"].min()) if "Crop_Year" in df.columns else None,
    "year_max": int(df["Crop_Year"].max()) if "Crop_Year" in df.columns else None,
    "yield_min": float(df["Yield"].min()) if "Yield" in df.columns else None,
    "yield_max": float(df["Yield"].max()) if "Yield" in df.columns else None,
    "yield_mean": float(df["Yield"].mean()) if "Yield" in df.columns else None,
}
with open(BASE_DIR / "outputs" / "audit_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nAudit summary saved to outputs/audit_summary.json")
