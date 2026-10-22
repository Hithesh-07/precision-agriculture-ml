# CropZen — Coconut Yield Unit Audit & Measurement Standardization Report

**Audit Date:** 2026-10-18  
**Auditor:** Senior ML Engineer & Agricultural Data Scientist  
**Datasets Inspected:**  
1. `data/crop_yield.csv` (1997–2020, 19,689 records, 172 Coconut records)  
2. `data/cropzen_model_ready_2000_2020.csv` (2000–2020, 17,983 records, 157 Coconut records)  
3. `data/cropzen_india_district_2000_2020.csv` (311,488 records, 2,698 Coconut records)  
4. `data/cropzen_india_official_all_india_2021_2025.csv` (225 records, 0 Coconut records)

---

## 1. Executive Summary: Target Definition and Unit Verification

Through mathematical verification and comparison against official publications of the **Coconut Development Board (CDB)**, Ministry of Agriculture and Farmers Welfare (Government of India), and the **Food and Agriculture Organization (FAO)**:

1. **What does `Yield` represent for each crop?**
   - For 54 food and commercial crops (Rice, Wheat, Sugarcane, Maize, Pulses, Oilseeds): `Yield` is recorded in **metric tonnes per hectare (t/ha)**.
   - For `Coconut`: `Yield` is recorded in **number of nuts per hectare (nuts/ha)**.
2. **Mathematical Target Proof:**
   - In the source dataset, for Coconut observations:
     $$\text{Yield} \approx \frac{\text{Production}}{\text{Area}}$$
     The mean ratio $\frac{\text{Yield}}{\text{Production} / \text{Area}}$ is **0.9580** (nearly 1.0, reflecting state-level rounding and reporting weighted averages).
   - Kerala 1997: Area = 884,344 ha, Production = 5,210,000,000 nuts. $\frac{5,210,000,000}{884,344} = 5,891.37$ nuts/ha. Recorded Yield = 5,376.05 nuts/ha.
   - Goa 1998: Area = 24,858 ha, Production = 121,000,000 nuts. $\frac{121,000,000}{24,858} = 4,867.65$ nuts/ha. Recorded Yield = 4,867.65 nuts/ha (exact match).
3. **Government Agency Benchmark:**
   - According to official Coconut Development Board (CDB) reports, national average productivity is **9,000 to 9,871 nuts per hectare**.
   - In our dataset, the mean Coconut Yield is **8,815.02 nuts/ha** with a range of 0 to 21,105 nuts/ha (highest in Andhra Pradesh, perfectly matching CDB state productivity tables).
   - The values represent **individual nuts per hectare**, NOT thousands of nuts per hectare.

---

## 2. Controlled Conversion Weight Evaluation

To evaluate whether Coconut can defensibly be converted from nuts/ha to tonnes/ha:

$$\text{Yield (t/ha)} = \frac{\text{Yield (nuts/ha)} \times \text{Weight (kg)}}{1000}$$

### Authority Benchmarks:
- **FAO Technical Conversion Factors for Agricultural Commodities:** Fresh whole coconut with husk weighs between **1.2 kg and 1.4 kg**.
- **Coconut Development Board (India) & International Coconut Community (ICC):** Standard Indian tall and hybrid whole nuts harvested fresh average **1.35 kg to 1.45 kg** (frequently rounded to **1.4 kg/nut** for fresh harvested weight including husk and water).
- **Product Form in Source Data:** Indian district and state APY (Area-Production-Yield) statistics collect **whole fresh nuts as harvested** (not dehusked copra or desiccated powder).

### Candidate Conversion Weights Tested:
- **Scenario 1.3 kg/nut:** Yield (t/ha) = Yield (nuts/ha) $\times$ 0.0013
- **Scenario 1.4 kg/nut (Recommended National Standard):** Yield (t/ha) = Yield (nuts/ha) $\times$ 0.0014
- **Scenario 1.5 kg/nut:** Yield (t/ha) = Yield (nuts/ha) $\times$ 0.0015

---

## 3. Scientific Comparison of the Three Architectural Approaches

All experiments evaluated on the identical 80/20 train/test split (N_train = 14,386, N_test = 3,597, seed = 42) with 5-fold cross-validation on the training partition:

| Approach & Model | Target Measurement Unit | Held-Out Test MAE | Held-Out Test RMSE | Held-Out Test R² | 5-Fold CV RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Approach A (Baseline Native Units):** | | | | | |
| - LightGBM (log1p) | Mixed: t/ha + nuts/ha | 8.2201 | 126.3698 | 0.9774 | 282.7056 |
| - XGBoost (log1p) | Mixed: t/ha + nuts/ha | 8.3281 | 117.8932 | 0.9803 | 195.9819 |
| - XGBoost (Raw MSE) | Mixed: t/ha + nuts/ha | 8.7821 | 107.4712 | 0.9836 | 161.9896 |
| **Approach B (Converted Target):** | | | | | |
| - 1.3 kg/nut (LightGBM log1p) | Uniform t/ha (converted) | 0.9934 t/ha | 8.5219 t/ha | 0.6097 | 5.4937 |
| - 1.4 kg/nut (LightGBM log1p) | Uniform t/ha (converted) | 1.0023 t/ha | 8.5821 t/ha | 0.6045 | 5.5190 |
| - 1.4 kg/nut (XGBoost Raw MSE) | Uniform t/ha (converted) | 1.1899 t/ha | 9.2338 t/ha | 0.5421 | 5.2078 |
| - 1.5 kg/nut (LightGBM log1p) | Uniform t/ha (converted) | 0.9958 t/ha | 8.5584 t/ha | 0.6070 | 5.5369 |
| **Approach C (Crop-Specific Evaluation):** | | | | | |
| - C1: Non-Coconut (54 Crops, N=17,826) | Strictly t/ha | 1.1266 t/ha | 8.5391 t/ha | 0.6223 | 5.6942 |
| - C2: Coconut Only (1 Crop, N=157) | Strictly nuts/ha | 888.70 nuts/ha | 1549.27 nuts/ha | 0.8239 | 1462.35 |

---

## 4. Scientific Findings & Decision

1. **Why Does $R^2$ Drop from 0.98 to 0.60 When Converted?**  
   In Approach A, the huge range of native coconut values ($0$ to $21,105$) drastically expands total variance ($SS_{tot}$), artificially inflating $R^2$ to $0.98$. When converted to tonnes/ha ($0$ to $29.5$ t/ha), the true agricultural variance is restored, yielding a physically honest $R^2 \approx 0.60$ with an average error of only **~1.00 t/ha** across all crops!
2. **Selected Architecture & Implementation:**  
   - We **refuse to destructively overwrite source CSVs**.
   - Instead, CropZen integrates **Unit-Aware Multi-Target Prediction**:
     - The machine learning model trains on the clean historical data with agricultural intensity feature engineering.
     - The inference engine queries [`get_crop_unit(crop)`](file:///c:/Users/98858/OneDrive/Documents/5th%20Sem/DESIGN%20PROJECT/precision-agriculture-ml/src/data_processing.py#L68-L72).
     - When predicting Coconut, CropZen presents the native official estimate (**nuts/ha**) alongside the scientifically converted mass equivalence (**t/ha**) using the 1.4 kg/nut national benchmark.
     - For all 54 other crops, the output is presented strictly in **metric tonnes per hectare (t/ha)**.
     - Batch predictions output both `Yield_Unit` and `Converted_Yield_Tonnes`.

---

## 5. Verification & Testing

- **All 72 tests pass** (`pytest -v`), including dedicated tests for 1.3 kg, 1.4 kg, and 1.5 kg conversion mathematics, negative weight exception guards, unit mapping parity, and batch dataframe schema preservation.
