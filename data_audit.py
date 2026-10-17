"""
data_audit.py
=============
Reproducible data-audit script for all CropZen datasets.

Run:
    .venv\\Scripts\\python.exe data_audit.py

Outputs:
    reports/data_audit.md  — human-readable audit report
    outputs/audit_summary.json — machine-readable summary

This script inspects all five datasets without modifying them.
It does NOT delete, impute, or alter any data.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    DATASET_REGISTRY,
    DATA_PATH,
    DATA_PATH_MODEL_READY,
    DATA_PATH_DISTRICT,
    DATA_PATH_OFFICIAL_2021_25,
    DATA_PATH_COMBINED,
)

REPORTS_DIR = Path("reports")
OUTPUTS_DIR = Path("outputs")
REPORTS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

DATASETS_TO_AUDIT = [
    ("original",          DATA_PATH),
    ("model_ready",       DATA_PATH_MODEL_READY),
    ("district",          DATA_PATH_DISTRICT),
    ("official_2021_25",  DATA_PATH_OFFICIAL_2021_25),
    ("combined",          DATA_PATH_COMBINED),
]

report_lines = [
    "# CropZen — Comprehensive Data Audit Report",
    f"*Script: data_audit.py — run with `.venv\\Scripts\\python.exe data_audit.py`*",
    "",
]

audit_summary = {}


def fmt_pct(n, total):
    if total == 0:
        return "N/A"
    return f"{n:,} ({n / total * 100:.1f}%)"


def audit_dataset(key: str, path: Path) -> dict:
    reg = DATASET_REGISTRY.get(key, {})
    lines = [
        f"",
        f"---",
        f"",
        f"## {reg.get('label', key)}",
        f"",
        f"**File:** `{path.name}` ({path.stat().st_size / 1_048_576:.2f} MB)" if path.exists() else "**File:** NOT FOUND",
        f"**Geography:** {reg.get('geography', 'Unknown')}",
        f"**Year coverage:** {reg.get('years', 'Unknown')}",
        f"**Model-ready:** {'✅ Yes' if reg.get('model_ready') else '❌ No'}",
        f"**Has observed Yield:** {'✅ Yes' if reg.get('has_yield') else '❌ No — yield validation impossible'}",
        f"",
    ]

    result = {"key": key, "path": str(path), "file_exists": path.exists()}

    if not path.exists():
        lines.append(f"> ⚠️ **File not found.** Skipping audit.")
        result["error"] = "file_not_found"
        return result, lines

    df = pd.read_csv(path, low_memory=False)

    n_rows, n_cols = df.shape
    n_dups = int(df.duplicated().sum())
    result.update({"rows": n_rows, "cols": n_cols, "duplicates": n_dups})

    lines += [
        f"### Shape",
        f"- **Rows:** {n_rows:,}",
        f"- **Columns:** {n_cols}",
        f"- **Duplicate rows:** {n_dups:,}",
        f"",
    ]

    # Column summary
    lines.append(f"### Column Summary")
    lines.append(f"")
    lines.append(f"| Column | Dtype | Nulls | Null % | Unique |")
    lines.append(f"|--------|-------|-------|--------|--------|")
    col_summary = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_null = int(df[col].isna().sum())
        pct_null = n_null / n_rows * 100 if n_rows else 0
        n_uniq = int(df[col].nunique())
        lines.append(f"| `{col}` | {dtype} | {n_null:,} | {pct_null:.1f}% | {n_uniq:,} |")
        col_summary[col] = {"dtype": dtype, "nulls": n_null, "null_pct": round(pct_null, 2), "unique": n_uniq}
    result["columns"] = col_summary
    lines.append(f"")

    # Year range
    year_col = next((c for c in ["Crop_Year", "crop_year"] if c in df.columns), None)
    if year_col:
        yvals = pd.to_numeric(df[year_col], errors="coerce").dropna()
        yr_min, yr_max = int(yvals.min()), int(yvals.max())
        result["year_min"] = yr_min
        result["year_max"] = yr_max
        lines.append(f"### Year Range")
        lines.append(f"- **`{year_col}`:** {yr_min} – {yr_max} ({yvals.nunique()} unique values)")
        lines.append(f"")

    # Numeric summary
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        desc = df[num_cols].describe().round(3)
        lines.append(f"### Numeric Summary")
        lines.append(f"")
        lines.append(f"```")
        lines.append(desc.to_string())
        lines.append(f"```")
        lines.append(f"")

    # Category value counts
    cat_cols = [c for c in ["Crop", "Season", "State", "District", "geography_level"] if c in df.columns]
    if cat_cols:
        lines.append(f"### Categorical Values")
        lines.append(f"")
        for cc in cat_cols:
            vals = sorted(df[cc].dropna().unique().tolist())
            sample = vals[:10]
            lines.append(f"**`{cc}`** ({len(vals)} unique):")
            lines.append(f"  Sample: `{', '.join(str(v) for v in sample)}`{'...' if len(vals) > 10 else ''}")
            lines.append(f"")

    # Zero / negative checks
    lines.append(f"### Zero / Negative Value Checks")
    lines.append(f"")
    check_cols = {c: df[c] for c in ["Area", "Production", "Yield"] if c in df.columns and pd.api.types.is_numeric_dtype(df[c])}
    if check_cols:
        lines.append(f"| Column | Zeros | Negatives | Nulls |")
        lines.append(f"|--------|-------|-----------|-------|")
        for col, series in check_cols.items():
            zeros = int((series == 0).sum())
            negs  = int((series < 0).sum())
            nulls = int(series.isna().sum())
            lines.append(f"| `{col}` | {zeros:,} | {negs:,} | {nulls:,} |")
        lines.append(f"")

    # Target availability
    lines.append(f"### Target Availability (`Yield`)")
    lines.append(f"")
    if "Yield" in df.columns:
        missing_yield = int(df["Yield"].isna().sum())
        pct_missing = missing_yield / n_rows * 100 if n_rows else 0
        lines.append(f"- **Missing:** {missing_yield:,} ({pct_missing:.1f}%)")
        if missing_yield < n_rows:
            valid = df["Yield"].dropna()
            lines.append(f"- **Min:** {valid.min():.4f}  **Median:** {valid.median():.4f}  **Max:** {valid.max():.4f}")
            lines.append(f"- **Zero-yield rows:** {int((valid == 0).sum()):,}")
        else:
            lines.append(f"- ⛔ **All Yield values are missing.** Supervised validation is NOT possible.")
        result["yield_missing"] = missing_yield
    else:
        lines.append(f"- ⛔ No `Yield` column found.")
        result["yield_missing"] = n_rows
    lines.append(f"")

    # Leakage check
    if "Production" in df.columns and "Area" in df.columns and "Yield" in df.columns:
        lines.append(f"### Data Leakage Check")
        lines.append(f"")
        sub = df[["Production", "Area", "Yield"]].dropna()
        if len(sub) > 0 and (sub["Area"] > 0).all():
            derived = sub["Production"] / sub["Area"]
            match = (derived - sub["Yield"]).abs() < 0.01
            pct = match.mean() * 100
            lines.append(f"- `Yield == Production / Area` (within 0.01 tolerance): **{pct:.1f}%** of non-null rows")
            if pct > 50:
                lines.append(f"- ⚠️ **Strong leakage signal:** Production is essentially a proxy for Yield on this dataset. "
                             f"Including Production as a feature would cause data leakage.")
            else:
                lines.append(f"- ⚠️ Partial match — Production likely uses different aggregation or units. "
                             f"Still excluded as a predictor since it is a post-harvest quantity.")
            result["leakage_pct"] = round(pct, 2)
        lines.append(f"")

    # Model-readiness check
    lines.append(f"### Model Training Compatibility")
    lines.append(f"")
    missing_features = [f for f in ALL_FEATURES if f not in df.columns or df[f].isna().all()]
    if missing_features:
        lines.append(f"- ❌ **Missing or all-null features:** `{', '.join(missing_features)}`")
        lines.append(f"- ❌ **Cannot be used for model training** without these features.")
    elif reg.get("model_ready"):
        lines.append(f"- ✅ All required model features are present and non-null.")
        lines.append(f"- ✅ **Compatible with existing training pipeline** (after whitespace stripping).")
    else:
        lines.append(f"- ⚠️ Feature columns present but dataset is not designated for model training.")
    lines.append(f"")

    # Limitations
    if reg.get("limitations"):
        lines.append(f"### Known Limitations")
        lines.append(f"")
        lines.append(f"> {reg['limitations']}")
        lines.append(f"")

    return result, lines


print("Running CropZen data audit...")
for key, path in DATASETS_TO_AUDIT:
    print(f"  Auditing: {key} ...")
    result, lines = audit_dataset(key, path)
    audit_summary[key] = result
    report_lines.extend(lines)

# Write report
report_path = REPORTS_DIR / "data_audit.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"Audit report written to: {report_path}")

# Write JSON summary
summary_path = OUTPUTS_DIR / "audit_summary.json"
with open(summary_path, "w") as f:
    json.dump(audit_summary, f, indent=2)
print(f"Audit summary (JSON) written to: {summary_path}")
print("Audit complete.")
