"""
inspect_data.py
Inspect dataset distributions safely.
"""
import pandas as pd

def main():
    df = pd.read_csv('data/cropzen_model_ready_2000_2020.csv')
    print("Years:", sorted(df['Crop_Year'].unique()))
    print("Crops count:", df['Crop'].nunique())
    print("Total rows:", len(df))
    coconuts = df[df['Crop'] == 'Coconut']
    print(f"Coconut rows: {len(coconuts)} | Yield min: {coconuts['Yield'].min()} | max: {coconuts['Yield'].max()} | mean: {coconuts['Yield'].mean():.2f}")
    non_coco = df[df['Crop'] != 'Coconut']
    print(f"Non-coconut rows: {len(non_coco)} | Yield min: {non_coco['Yield'].min()} | max: {non_coco['Yield'].max()} | mean: {non_coco['Yield'].mean():.2f}")

if __name__ == '__main__':
    main()
