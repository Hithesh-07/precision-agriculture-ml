"""
inspect_crops.py
Check exact crop names with extreme yield.
"""
import pandas as pd

def main():
    df = pd.read_csv('data/cropzen_model_ready_2000_2020.csv')
    high_yield = df[df['Yield'] > 100]
    print("Crops with Yield > 100:")
    print(high_yield['Crop'].value_counts())
    print("\nSample Coconut matching:")
    print([c for c in df['Crop'].unique() if 'coco' in c.lower() or 'nut' in c.lower()])

if __name__ == '__main__':
    main()
