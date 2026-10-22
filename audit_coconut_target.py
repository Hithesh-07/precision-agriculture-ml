"""
audit_coconut_target.py
Inspect the actual source datasets (data/crop_yield.csv, data/cropzen_model_ready_2000_2020.csv,
data/cropzen_india_district_2000_2020.csv, data/cropzen_india_official_all_india_2021_2025.csv)
to understand the exact definition, scale, and relationship of coconut Area, Production, and Yield.
"""
import pandas as pd
import numpy as np

def audit():
    print("=== AUDITING DATA/CROP_YIELD.CSV ===")
    df_orig = pd.read_csv("data/crop_yield.csv")
    print(f"Columns: {list(df_orig.columns)}")
    coco_orig = df_orig[df_orig["Crop"].str.strip() == "Coconut"]
    print(f"Coconut rows: {len(coco_orig)}")
    if len(coco_orig) > 0:
        print("Coconut summary:")
        print(coco_orig[["Area", "Production", "Yield"]].describe().round(2))
        # Check Yield == Production / Area
        calculated_yield = coco_orig["Production"] / coco_orig["Area"]
        ratio_diff = (calculated_yield - coco_orig["Yield"]).abs()
        print(f"Max absolute diff between (Production/Area) and Yield: {ratio_diff.max():.4f}")
        print(f"Mean absolute diff: {ratio_diff.mean():.4f}")
        print("Sample 5 coconut records:")
        print(coco_orig[["Crop_Year", "State", "Area", "Production", "Yield"]].head(5))

    print("\n=== AUDITING DATA/CROPZEN_MODEL_READY_2000_2020.CSV ===")
    df_mr = pd.read_csv("data/cropzen_model_ready_2000_2020.csv")
    coco_mr = df_mr[df_mr["Crop"].str.strip() == "Coconut"]
    print(f"Coconut rows: {len(coco_mr)}")
    if len(coco_mr) > 0:
        print(coco_mr[["Area", "Production", "Yield"]].describe().round(2))
        calc = coco_mr["Production"] / coco_mr["Area"]
        print(f"Max diff (Production/Area - Yield): {(calc - coco_mr['Yield']).abs().max():.4f}")
        print("Sample 5 records:")
        print(coco_mr[["Crop_Year", "State", "Area", "Production", "Yield"]].head(5))

    print("\n=== AUDITING DISTRICT DATASET ===")
    df_dist = pd.read_csv("data/cropzen_india_district_2000_2020.csv")
    coco_dist = df_dist[df_dist["Crop"].str.strip() == "Coconut"]
    print(f"District coconut rows: {len(coco_dist)}")
    if "production_unit" in df_dist.columns or "yield_unit" in df_dist.columns:
        print("Units in district data:")
        for col in ["production_unit", "yield_unit"]:
            if col in df_dist.columns:
                print(col, coco_dist[col].value_counts())
    print("District coconut sample:")
    print(coco_dist.head(3))

    print("\n=== AUDITING OFFICIAL ALL INDIA 2021-2025 ===")
    df_off = pd.read_csv("data/cropzen_india_official_all_india_2021_2025.csv")
    coco_off = df_off[df_off["Crop"].str.strip().str.contains("Coconut", case=False, na=False)]
    print(f"Official coconut rows: {len(coco_off)}")
    if len(coco_off) > 0:
        print(coco_off.head(5))

if __name__ == "__main__":
    audit()
