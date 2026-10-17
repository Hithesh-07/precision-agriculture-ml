# CropZen — Comprehensive Data Audit Report
*Script: data_audit.py — run with `.venv\Scripts\python.exe data_audit.py`*


---

## Original Historical (1997–2020)

**File:** `crop_yield.csv` (1.55 MB)
**Geography:** State-level
**Year coverage:** 1997–2020
**Model-ready:** ✅ Yes
**Has observed Yield:** ✅ Yes

### Shape
- **Rows:** 19,689
- **Columns:** 10
- **Duplicate rows:** 0

### Column Summary

| Column | Dtype | Nulls | Null % | Unique |
|--------|-------|-------|--------|--------|
| `Crop` | str | 0 | 0.0% | 55 |
| `Crop_Year` | int64 | 0 | 0.0% | 24 |
| `Season` | str | 0 | 0.0% | 6 |
| `State` | str | 0 | 0.0% | 30 |
| `Area` | float64 | 0 | 0.0% | 13,644 |
| `Production` | int64 | 0 | 0.0% | 14,016 |
| `Annual_Rainfall` | float64 | 0 | 0.0% | 634 |
| `Fertilizer` | float64 | 0 | 0.0% | 18,598 |
| `Pesticide` | float64 | 0 | 0.0% | 17,405 |
| `Yield` | float64 | 0 | 0.0% | 13,551 |

### Year Range
- **`Crop_Year`:** 1997 – 2020 (24 unique values)

### Numeric Summary

```
       Crop_Year          Area    Production  Annual_Rainfall    Fertilizer     Pesticide      Yield
count  19689.000  1.968900e+04  1.968900e+04        19689.000  1.968900e+04  1.968900e+04  19689.000
mean    2009.128  1.799266e+05  1.643594e+07         1437.755  2.410331e+07  4.884835e+04     79.954
std        6.498  7.328287e+05  2.630568e+08          816.910  9.494600e+07  2.132874e+05    878.306
min     1997.000  5.000000e-01  0.000000e+00          301.300  5.417000e+01  9.000000e-02      0.000
25%     2004.000  1.390000e+03  1.393000e+03          940.700  1.880146e+05  3.567000e+02      0.600
50%     2010.000  9.317000e+03  1.380400e+04         1247.600  1.234957e+06  2.421900e+03      1.030
75%     2015.000  7.511200e+04  1.227180e+05         1643.700  1.000385e+07  2.004170e+04      2.389
max     2020.000  5.080810e+07  6.326000e+09         6552.700  4.835407e+09  1.575051e+07  21105.000
```

### Categorical Values

**`Crop`** (55 unique):
  Sample: `Arecanut, Arhar/Tur, Bajra, Banana, Barley, Black pepper, Cardamom, Cashewnut, Castor seed, Coconut `...

**`Season`** (6 unique):
  Sample: `Autumn     , Kharif     , Rabi       , Summer     , Whole Year , Winter     `

**`State`** (30 unique):
  Sample: `Andhra Pradesh, Arunachal Pradesh, Assam, Bihar, Chhattisgarh, Delhi, Goa, Gujarat, Haryana, Himachal Pradesh`...

### Zero / Negative Value Checks

| Column | Zeros | Negatives | Nulls |
|--------|-------|-----------|-------|
| `Area` | 0 | 0 | 0 |
| `Production` | 112 | 0 | 0 |
| `Yield` | 112 | 0 | 0 |

### Target Availability (`Yield`)

- **Missing:** 0 (0.0%)
- **Min:** 0.0000  **Median:** 1.0300  **Max:** 21105.0000
- **Zero-yield rows:** 112

### Data Leakage Check

- `Yield == Production / Area` (within 0.01 tolerance): **21.3%** of non-null rows
- ⚠️ Partial match — Production likely uses different aggregation or units. Still excluded as a predictor since it is a post-harvest quantity.

### Model Training Compatibility

- ✅ All required model features are present and non-null.
- ✅ **Compatible with existing training pipeline** (after whitespace stripping).

### Known Limitations

> Random 80/20 split may allow (crop, state) overlap between train and test.


---

## Model-Ready Curated (2000–2020)

**File:** `cropzen_model_ready_2000_2020.csv` (1.44 MB)
**Geography:** State-level
**Year coverage:** 2000–2020
**Model-ready:** ✅ Yes
**Has observed Yield:** ✅ Yes

### Shape
- **Rows:** 17,983
- **Columns:** 10
- **Duplicate rows:** 0

### Column Summary

| Column | Dtype | Nulls | Null % | Unique |
|--------|-------|-------|--------|--------|
| `Crop` | str | 0 | 0.0% | 55 |
| `Crop_Year` | int64 | 0 | 0.0% | 21 |
| `Season` | str | 0 | 0.0% | 6 |
| `State` | str | 0 | 0.0% | 30 |
| `Area` | float64 | 0 | 0.0% | 12,585 |
| `Production` | int64 | 0 | 0.0% | 12,930 |
| `Annual_Rainfall` | float64 | 0 | 0.0% | 566 |
| `Fertilizer` | float64 | 0 | 0.0% | 16,942 |
| `Pesticide` | float64 | 0 | 0.0% | 15,883 |
| `Yield` | float64 | 0 | 0.0% | 12,506 |

### Year Range
- **`Crop_Year`:** 2000 – 2020 (21 unique values)

### Numeric Summary

```
       Crop_Year          Area    Production  Annual_Rainfall    Fertilizer    Pesticide      Yield
count   17983.00  1.798300e+04  1.798300e+04        17983.000  1.798300e+04    17983.000  17983.000
mean     2010.17  1.712353e+05  1.674371e+07         1416.969  2.381838e+07    45990.801     81.292
std         5.80  6.275418e+05  2.658686e+08          778.566  8.981196e+07   178228.483    890.042
min      2000.00  5.000000e-01  0.000000e+00          301.300  5.417000e+01        0.090      0.000
25%      2005.00  1.317500e+03  1.317000e+03          940.700  1.826757e+05      337.750      0.603
50%      2011.00  8.762000e+03  1.291100e+04         1235.600  1.180800e+06     2283.270      1.035
75%      2015.00  6.960000e+04  1.158295e+05         1631.400  9.619963e+06    18291.075      2.427
max      2020.00  1.021652e+07  6.326000e+09         6258.800  1.754789e+09  3780111.290  21105.000
```

### Categorical Values

**`Crop`** (55 unique):
  Sample: `Arecanut, Arhar/Tur, Bajra, Banana, Barley, Black pepper, Cardamom, Cashewnut, Castor seed, Coconut `...

**`Season`** (6 unique):
  Sample: `Autumn     , Kharif     , Rabi       , Summer     , Whole Year , Winter     `

**`State`** (30 unique):
  Sample: `Andhra Pradesh, Arunachal Pradesh, Assam, Bihar, Chhattisgarh, Delhi, Goa, Gujarat, Haryana, Himachal Pradesh`...

### Zero / Negative Value Checks

| Column | Zeros | Negatives | Nulls |
|--------|-------|-----------|-------|
| `Area` | 0 | 0 | 0 |
| `Production` | 110 | 0 | 0 |
| `Yield` | 108 | 0 | 0 |

### Target Availability (`Yield`)

- **Missing:** 0 (0.0%)
- **Min:** 0.0000  **Median:** 1.0355  **Max:** 21105.0000
- **Zero-yield rows:** 108

### Data Leakage Check

- `Yield == Production / Area` (within 0.01 tolerance): **21.4%** of non-null rows
- ⚠️ Partial match — Production likely uses different aggregation or units. Still excluded as a predictor since it is a post-harvest quantity.

### Model Training Compatibility

- ✅ All required model features are present and non-null.
- ✅ **Compatible with existing training pipeline** (after whitespace stripping).

### Known Limitations

> Covers 2000–2020 only; excludes 1997–1999 records present in the original dataset.


---

## District-Level Historical (2000–2020)

**File:** `cropzen_india_district_2000_2020.csv` (123.45 MB)
**Geography:** District-level (706 districts, 37 states)
**Year coverage:** 2000–2020
**Model-ready:** ❌ No
**Has observed Yield:** ✅ Yes

### Shape
- **Rows:** 311,488
- **Columns:** 23
- **Duplicate rows:** 0

### Column Summary

| Column | Dtype | Nulls | Null % | Unique |
|--------|-------|-------|--------|--------|
| `State` | str | 0 | 0.0% | 37 |
| `District` | str | 0 | 0.0% | 706 |
| `Crop` | str | 5 | 0.0% | 55 |
| `Season` | str | 0 | 0.0% | 6 |
| `Crop_Year` | int64 | 0 | 0.0% | 21 |
| `crop_year_label` | int64 | 0 | 0.0% | 21 |
| `crop_year_start` | int64 | 0 | 0.0% | 21 |
| `crop_year_end` | int64 | 0 | 0.0% | 21 |
| `Area` | float64 | 0 | 0.0% | 44,850 |
| `Production` | float64 | 4,598 | 1.5% | 56,398 |
| `Yield` | float64 | 0 | 0.0% | 10,139 |
| `Annual_Rainfall` | float64 | 311,488 | 100.0% | 0 |
| `Fertilizer` | float64 | 311,488 | 100.0% | 0 |
| `Pesticide` | float64 | 311,488 | 100.0% | 0 |
| `geography_level` | str | 0 | 0.0% | 1 |
| `record_type` | str | 0 | 0.0% | 1 |
| `estimate_status` | str | 0 | 0.0% | 1 |
| `area_unit` | str | 0 | 0.0% | 1 |
| `production_unit` | str | 0 | 0.0% | 1 |
| `yield_unit` | str | 0 | 0.0% | 1 |
| `source` | str | 0 | 0.0% | 1 |
| `source_url` | str | 0 | 0.0% | 1 |
| `source_notes` | str | 0 | 0.0% | 1 |

### Year Range
- **`Crop_Year`:** 2000 – 2020 (21 unique values)

### Numeric Summary

```
        Crop_Year  crop_year_label  crop_year_start  crop_year_end         Area    Production       Yield  Annual_Rainfall  Fertilizer  Pesticide
count  311488.000       311488.000       311488.000     311488.000   311488.000  3.068900e+05  311488.000              0.0         0.0        0.0
mean     2010.056         2010.056         2010.056       2010.056    11284.667  9.895822e+05      82.323              NaN         NaN        NaN
std         5.811            5.811            5.811          5.811    37821.524  2.195466e+07     937.530              NaN         NaN        NaN
min      2000.000         2000.000         2000.000       2000.000        0.004  0.000000e+00       0.000              NaN         NaN        NaN
25%      2005.000         2005.000         2005.000       2005.000       70.000  8.300000e+01       0.550              NaN         NaN        NaN
50%      2010.000         2010.000         2010.000       2010.000      500.000  6.930000e+02       1.010              NaN         NaN        NaN
75%      2015.000         2015.000         2015.000       2015.000     3900.000  6.964000e+03       2.520              NaN         NaN        NaN
max      2020.000         2020.000         2020.000       2020.000  1133397.000  1.597800e+09   43958.330              NaN         NaN        NaN
```

### Categorical Values

**`Crop`** (55 unique):
  Sample: `Arecanut, Arhar/Tur, Bajra, Banana, Barley, Black pepper, Cardamom, Cashewnut, Castor seed, Coconut`...

**`Season`** (6 unique):
  Sample: `Autumn, Kharif, Rabi, Summer, Whole Year, Winter`

**`State`** (37 unique):
  Sample: `Andaman and Nicobar Island, Andhra Pradesh, Arunachal Pradesh, Assam, Bihar, CHANDIGARH, Chhattisgarh, Dadra and Nagar Haveli, Daman and Diu, Delhi`...

**`District`** (706 unique):
  Sample: `ADILABAD, AGAR MALWA, AGRA, AHMADABAD, AHMEDNAGAR, AIZAWL, AJMER, AKOLA, ALAPPUZHA, ALIGARH`...

**`geography_level`** (1 unique):
  Sample: `district`

### Zero / Negative Value Checks

| Column | Zeros | Negatives | Nulls |
|--------|-------|-----------|-------|
| `Area` | 0 | 0 | 0 |
| `Production` | 1,301 | 0 | 4,598 |
| `Yield` | 5,566 | 0 | 0 |

### Target Availability (`Yield`)

- **Missing:** 0 (0.0%)
- **Min:** 0.0000  **Median:** 1.0100  **Max:** 43958.3300
- **Zero-yield rows:** 5,566

### Data Leakage Check

- `Yield == Production / Area` (within 0.01 tolerance): **98.8%** of non-null rows
- ⚠️ **Strong leakage signal:** Production is essentially a proxy for Yield on this dataset. Including Production as a feature would cause data leakage.

### Model Training Compatibility

- ❌ **Missing or all-null features:** `Annual_Rainfall, Fertilizer, Pesticide`
- ❌ **Cannot be used for model training** without these features.

### Known Limitations

> Annual_Rainfall, Fertilizer, Pesticide are 100% missing — cannot train existing model.


---

## Official All-India 2020-21 to 2024-25

**File:** `cropzen_india_official_all_india_2021_2025.csv` (0.07 MB)
**Geography:** All-India aggregate
**Year coverage:** 2020-21 to 2024-25
**Model-ready:** ❌ No
**Has observed Yield:** ❌ No — yield validation impossible

### Shape
- **Rows:** 225
- **Columns:** 23
- **Duplicate rows:** 0

### Column Summary

| Column | Dtype | Nulls | Null % | Unique |
|--------|-------|-------|--------|--------|
| `State` | str | 0 | 0.0% | 1 |
| `District` | float64 | 225 | 100.0% | 0 |
| `Crop` | str | 0 | 0.0% | 27 |
| `Season` | str | 0 | 0.0% | 3 |
| `Crop_Year` | int64 | 0 | 0.0% | 5 |
| `crop_year_label` | str | 0 | 0.0% | 5 |
| `crop_year_start` | int64 | 0 | 0.0% | 5 |
| `crop_year_end` | int64 | 0 | 0.0% | 5 |
| `Area` | float64 | 225 | 100.0% | 0 |
| `Production` | float64 | 0 | 0.0% | 199 |
| `Yield` | float64 | 225 | 100.0% | 0 |
| `Annual_Rainfall` | float64 | 225 | 100.0% | 0 |
| `Fertilizer` | float64 | 225 | 100.0% | 0 |
| `Pesticide` | float64 | 225 | 100.0% | 0 |
| `geography_level` | str | 0 | 0.0% | 1 |
| `record_type` | str | 0 | 0.0% | 1 |
| `estimate_status` | str | 0 | 0.0% | 2 |
| `area_unit` | float64 | 225 | 100.0% | 0 |
| `production_unit` | str | 0 | 0.0% | 3 |
| `yield_unit` | float64 | 225 | 100.0% | 0 |
| `source` | str | 0 | 0.0% | 1 |
| `source_url` | str | 0 | 0.0% | 1 |
| `source_notes` | str | 0 | 0.0% | 1 |

### Year Range
- **`Crop_Year`:** 2021 – 2025 (5 unique values)

### Numeric Summary

```
       District  Crop_Year  crop_year_start  crop_year_end  Area  Production  Yield  Annual_Rainfall  Fertilizer  Pesticide  area_unit  yield_unit
count       0.0    225.000          225.000        225.000   0.0     225.000    0.0              0.0         0.0        0.0        0.0         0.0
mean        NaN   2023.000         2022.000       2023.000   NaN     190.625    NaN              NaN         NaN        NaN        NaN         NaN
std         NaN      1.417            1.417          1.417   NaN     689.358    NaN              NaN         NaN        NaN        NaN         NaN
min         NaN   2021.000         2020.000       2021.000   NaN       0.000    NaN              NaN         NaN        NaN        NaN         NaN
25%         NaN   2022.000         2021.000       2022.000   NaN       2.040    NaN              NaN         NaN        NaN        NaN         NaN
50%         NaN   2023.000         2022.000       2023.000   NaN      16.470    NaN              NaN         NaN        NaN        NaN         NaN
75%         NaN   2024.000         2023.000       2024.000   NaN      98.480    NaN              NaN         NaN        NaN        NaN         NaN
max         NaN   2025.000         2024.000       2025.000   NaN    4905.330    NaN              NaN         NaN        NaN        NaN         NaN
```

### Categorical Values

**`Crop`** (27 unique):
  Sample: `Bajra, Barley, Castorseed, Cotton, Gram, Groundnut, Jowar, Jute, Lentil, Linseed`...

**`Season`** (3 unique):
  Sample: `Kharif, Rabi, Summer`

**`State`** (1 unique):
  Sample: `All India`

**`District`** (0 unique):
  Sample: ``

**`geography_level`** (1 unique):
  Sample: `all_india`

### Zero / Negative Value Checks

| Column | Zeros | Negatives | Nulls |
|--------|-------|-----------|-------|
| `Area` | 0 | 0 | 225 |
| `Production` | 20 | 0 | 0 |
| `Yield` | 0 | 0 | 225 |

### Target Availability (`Yield`)

- **Missing:** 225 (100.0%)
- ⛔ **All Yield values are missing.** Supervised validation is NOT possible.

### Data Leakage Check


### Model Training Compatibility

- ❌ **Missing or all-null features:** `Area, Annual_Rainfall, Fertilizer, Pesticide`
- ❌ **Cannot be used for model training** without these features.

### Known Limitations

> Yield is 100% missing. Area is 100% missing. Supervised model validation is not possible. Production in mixed units (lakh tonnes / lakh bales). Cannot be compared directly to state-level predictions.


---

## Combined Dataset (2000–2025)

**File:** `cropzen_india_combined_2000_2025.csv` (123.52 MB)
**Geography:** Mixed: district + all-India
**Year coverage:** 2000–2025
**Model-ready:** ❌ No
**Has observed Yield:** ✅ Yes

### Shape
- **Rows:** 311,713
- **Columns:** 23
- **Duplicate rows:** 0

### Column Summary

| Column | Dtype | Nulls | Null % | Unique |
|--------|-------|-------|--------|--------|
| `State` | str | 0 | 0.0% | 38 |
| `District` | str | 225 | 0.1% | 706 |
| `Crop` | str | 5 | 0.0% | 65 |
| `Season` | str | 0 | 0.0% | 6 |
| `Crop_Year` | int64 | 0 | 0.0% | 26 |
| `crop_year_label` | str | 0 | 0.0% | 26 |
| `crop_year_start` | int64 | 0 | 0.0% | 25 |
| `crop_year_end` | int64 | 0 | 0.0% | 26 |
| `Area` | float64 | 225 | 0.1% | 44,850 |
| `Production` | float64 | 4,598 | 1.5% | 56,596 |
| `Yield` | float64 | 225 | 0.1% | 10,139 |
| `Annual_Rainfall` | float64 | 311,713 | 100.0% | 0 |
| `Fertilizer` | float64 | 311,713 | 100.0% | 0 |
| `Pesticide` | float64 | 311,713 | 100.0% | 0 |
| `geography_level` | str | 0 | 0.0% | 2 |
| `record_type` | str | 0 | 0.0% | 2 |
| `estimate_status` | str | 0 | 0.0% | 2 |
| `area_unit` | str | 225 | 0.1% | 1 |
| `production_unit` | str | 0 | 0.0% | 4 |
| `yield_unit` | str | 225 | 0.1% | 1 |
| `source` | str | 0 | 0.0% | 2 |
| `source_url` | str | 0 | 0.0% | 2 |
| `source_notes` | str | 0 | 0.0% | 2 |

### Year Range
- **`Crop_Year`:** 2000 – 2025 (26 unique values)

### Numeric Summary

```
        Crop_Year  crop_year_start  crop_year_end         Area    Production       Yield  Annual_Rainfall  Fertilizer  Pesticide
count  311713.000       311713.000     311713.000   311488.000  3.071150e+05  311488.000              0.0         0.0        0.0
mean     2010.066         2010.065       2010.066    11284.667  9.888574e+05      82.323              NaN         NaN        NaN
std         5.820            5.818          5.820    37821.524  2.194664e+07     937.530              NaN         NaN        NaN
min      2000.000         2000.000       2000.000        0.004  0.000000e+00       0.000              NaN         NaN        NaN
25%      2005.000         2005.000       2005.000       70.000  8.300000e+01       0.550              NaN         NaN        NaN
50%      2010.000         2010.000       2010.000      500.000  6.910000e+02       1.010              NaN         NaN        NaN
75%      2015.000         2015.000       2015.000     3900.000  6.950000e+03       2.520              NaN         NaN        NaN
max      2025.000         2024.000       2025.000  1133397.000  1.597800e+09   43958.330              NaN         NaN        NaN
```

### Categorical Values

**`Crop`** (65 unique):
  Sample: `Arecanut, Arhar/Tur, Bajra, Banana, Barley, Black pepper, Cardamom, Cashewnut, Castor seed, Castorseed`...

**`Season`** (6 unique):
  Sample: `Autumn, Kharif, Rabi, Summer, Whole Year, Winter`

**`State`** (38 unique):
  Sample: `All India, Andaman and Nicobar Island, Andhra Pradesh, Arunachal Pradesh, Assam, Bihar, CHANDIGARH, Chhattisgarh, Dadra and Nagar Haveli, Daman and Diu`...

**`District`** (706 unique):
  Sample: `ADILABAD, AGAR MALWA, AGRA, AHMADABAD, AHMEDNAGAR, AIZAWL, AJMER, AKOLA, ALAPPUZHA, ALIGARH`...

**`geography_level`** (2 unique):
  Sample: `all_india, district`

### Zero / Negative Value Checks

| Column | Zeros | Negatives | Nulls |
|--------|-------|-----------|-------|
| `Area` | 0 | 0 | 225 |
| `Production` | 1,321 | 0 | 4,598 |
| `Yield` | 5,566 | 0 | 225 |

### Target Availability (`Yield`)

- **Missing:** 225 (0.1%)
- **Min:** 0.0000  **Median:** 1.0100  **Max:** 43958.3300
- **Zero-yield rows:** 5,566

### Data Leakage Check

- `Yield == Production / Area` (within 0.01 tolerance): **98.8%** of non-null rows
- ⚠️ **Strong leakage signal:** Production is essentially a proxy for Yield on this dataset. Including Production as a feature would cause data leakage.

### Model Training Compatibility

- ❌ **Missing or all-null features:** `Annual_Rainfall, Fertilizer, Pesticide`
- ❌ **Cannot be used for model training** without these features.

### Known Limitations

> Mixes district and all-India geographic levels. Annual_Rainfall, Fertilizer, Pesticide 100% missing. Always filter by geography_level before analysis.
