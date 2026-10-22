"""
deep_audit_coconut.py
Investigate the exact production numbers and yield calculations for Coconut.
"""
import pandas as pd

df = pd.read_csv("data/crop_yield.csv")
coco = df[df["Crop"].str.strip() == "Coconut"].copy()
coco["Prod_per_Area"] = coco["Production"] / coco["Area"]
coco["Ratio"] = coco["Yield"] / coco["Prod_per_Area"]

print("Sample records comparing Yield and Production/Area:")
cols = ["Crop_Year", "State", "Area", "Production", "Yield", "Prod_per_Area", "Ratio"]
print(coco[cols].head(10))

print("\nDistribution of Ratio (Yield / (Production/Area)):")
print(coco["Ratio"].describe())

# Check Rice for comparison
rice = df[df["Crop"].str.strip() == "Rice"].copy()
rice["Prod_per_Area"] = rice["Production"] / rice["Area"]
rice["Ratio"] = rice["Yield"] / rice["Prod_per_Area"]
print("\nDistribution of Ratio for Rice:")
print(rice["Ratio"].describe())
