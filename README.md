# CropZen — AI-Powered Precision Agriculture and Crop Yield Optimization System

Live Deployment: [https://cropzen.streamlit.app/](https://cropzen.streamlit.app/)

An end-to-end Machine Learning web application designed to analyze historical agricultural records across Indian states, train multiple regression models, and estimate crop yields per hectare based on regional, meteorological, and farming inputs.

---

## 📌 Problem Statement & Scope

Crop yield prediction is inherently challenging because agricultural production varies widely with environmental conditions, crop characteristics, and regional differences. The reliability of machine-learning predictions depends on data quality, feature selection, and model performance.

This project implements a web-based system that:
1. Preprocesses historical Indian agricultural data (1997–2020).
2. Trains and compares **Linear Regression**, **Decision Tree Regressor**, and **Random Forest Regressor** pipelines.
3. Evaluates predictions on a held-out test split using standard regression metrics (**MAE**, **RMSE**, **R²**).
4. Provides interactive single-record predictions and batch CSV predictions.
5. Features full **bilingual English/Hindi interface** support and browser-based **speech interaction** (voice input and spoken synthesis).
6. Explains model behaviour via feature importance without asserting false causality.

> **Disclaimer:** This application is an educational prediction and analysis prototype developed for a 5th-semester CSE Design Project. It is **not** a certified farming advisory system and predictions should not be considered guaranteed outcomes.

---

## 📊 Dataset Description & Data Audit Findings

- **Location:** `data/crop_yield.csv` (19,689 records, 10 columns)
- **Timeframe:** 1997 – 2020 (24 years)
- **Coverage:** 55 unique crops, 30 Indian states, 6 agricultural seasons (*Kharif, Rabi, Whole Year, Summer, Autumn, Winter*)
- **Missing Values:** 0 missing values across all columns.
- **Duplicates:** 0 exact duplicate rows.

### Data Leakage Resolution
- **Investigation:** In raw agricultural statistics, `Production` represents the aggregate volume harvested. In our data leakage audit, `Yield` matches `Production / Area` in only 5.7% of rows (due to unit variations, e.g., coconuts counted in nuts vs. grains in tonnes).
- **Leakage Prevention Decision:** Since `Production` is an outcome measured post-harvest, including it would cause severe target leakage or trivial identity fitting. Therefore, **`Production` is strictly excluded from input features**.
- **Model Features:**
  - *Categorical:* `Crop`, `Season`, `State`
  - *Numerical:* `Crop_Year`, `Area` (ha), `Annual_Rainfall` (mm), `Fertilizer` (kg), `Pesticide` (kg)
- **Target Variable:** `Yield` (log1p transformed during training due to extreme right skewness, max 21,105 vs median 1.03; inverse `expm1` applied to output predictions).

---

## 🧠 Model Training & Evaluation Results

All models were evaluated on the **same held-out 20% test split (3,938 records)** with `random_state=42`. Preprocessing is embedded inside scikit-learn `Pipeline` objects using `ColumnTransformer` (OrdinalEncoder for categorical variables, StandardScaler for numeric variables).

| Model | MAE (tonnes/ha) | RMSE (tonnes/ha) | R² Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 78.7675 | 898.1971 | -0.0069 | Baseline |
| **Decision Tree Regressor** | **15.4667** | **297.1727** | **0.8898** | ⭐ **Selected Best Model** |
| **Random Forest Regressor** | 22.8233 | 395.7511 | 0.8045 | Ensemble Candidate |

*Note: R² measures the proportion of variance explained by the model (upper bounded by 1.0, can be negative for models performing worse than horizontal mean). It is not classification accuracy.*

---

## ⚙️ Setup and Installation (Windows / PowerShell)

### 1. Prerequisites
- Windows 10/11
- Python 3.13+ or Python 3.15 installed

### 2. Virtual Environment Setup
Open PowerShell in the project directory:
```powershell
cd "c:\Users\98858\OneDrive\Documents\5th Sem\DESIGN PROJECT\precision-agriculture-ml"

# Create virtual environment if not already created
python -m venv .venv

# Bootstrap pip if needed
.venv\Scripts\python.exe -m ensurepip --upgrade

# Install required dependencies
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 🚀 Execution Commands

### 1. Run Data Audit
To inspect data quality, verify schema integrity, and check leakage rules:
```powershell
.venv\Scripts\python.exe data_audit.py
```
Outputs audit statistics to console and saves `outputs/audit_summary.json`.

### 2. Train / Retrain Machine Learning Models
To fit the preprocessors, train all three regression algorithms, compare metrics on held-out data, and save artifacts:
```powershell
.venv\Scripts\python.exe train_model.py
```
Saves:
- `models/best_model.pkl`: Serialized model pipeline
- `outputs/feature_schema.json`: Categories, numeric bounds, and validation rules
- `outputs/evaluation.json`: Test metrics, split metadata, and test residuals

### 3. Run Automated Tests
```powershell
.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -v
```
*(All 42 unit and integration tests covering data loading, schema validation, cleaning, prediction, translations, out-of-range detection, empty/corrupt CSV validation, and duplicate header detection).*

### 4. Launch the Web Application
```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```
The application will launch and open in your default browser at `http://localhost:8501`.

---

## 🌐 Voice Interaction & Bilingual Support

- **English & Hindi Support:**
  - Fully translated interface accessible via the sidebar language selector (`English` / `हिन्दी`).
  - Translations stored centrally in `src/translations.py` with fallback mechanisms.
- **Voice Input (Speech-to-Text):**
  - Integrated using the browser Web Speech API (`SpeechRecognition`).
  - Supports English (`en-IN`) and Hindi (`hi-IN`). Supported in Google Chrome and Microsoft Edge.
  - Form fields remain fully editable by typing. User reviews recognized speech before applying it.
- **Spoken Output (Text-to-Speech):**
  - Uses browser `SpeechSynthesis`.
  - Reads predicted yield, unit, and model summary in the active language.
  - No external paid APIs or credentials required; privacy-friendly.

---

## 🛡️ Security & Reliability Controls

1. **Retraining Guard:** The web application disables arbitrary public retraining to prevent resource exhaustion and denial-of-service. Retraining is executed explicitly via `train_model.py`.
2. **Safe Model Deserialization:** The app loads models strictly from project-controlled paths (`models/best_model.pkl`), never from user uploads.
3. **Batch File Validation:** Uploaded CSVs are subject to a 5 MB file size limit, 5,000-row batch limit, strict column validation, and data type sanitization. Code execution or formulas in CSVs are rejected.
4. **Session Privacy:** Prediction history is transiently stored in Streamlit `session_state` and never persisted to external databases.

---

## 📁 Project Architecture

```
precision-agriculture-ml/
├── app.py                     # Main Streamlit web application
├── train_model.py             # Reproducible training & model comparison pipeline
├── data_audit.py              # Exploratory data audit & leakage verification
├── requirements.txt           # Verified Python package dependencies
├── README.md                  # Comprehensive documentation
├── .gitignore                 # Artifact & environment exclusions
├── data/
│   └── crop_yield.csv         # Original historical agricultural dataset
├── models/
│   └── best_model.pkl         # Saved production pipeline (Decision Tree Regressor)
├── outputs/
│   ├── audit_summary.json     # Data audit summary
│   ├── evaluation.json        # Test evaluation metrics & residuals
│   └── feature_schema.json    # Validated category maps & feature ranges
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Data cleaning, schema validation & ColumnTransformer
│   ├── model_training.py      # Model training logic, log1p transformation & evaluation
│   ├── prediction.py          # Single-record & batch CSV inference engine
│   ├── translations.py        # Centralized English & Hindi translations
│   └── voice_utils.py         # Browser Web Speech API components
└── tests/
    ├── __init__.py
    └── test_pipeline.py       # Pytest automated test suite (35 tests passing)
```

---

## ☁️ Deployment Instructions (Streamlit Community Cloud)

1. Push this repository to GitHub (ensure `models/best_model.pkl` and `outputs/` are committed).
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub repository.
4. Set Main file path to `app.py`.
5. Deploy. The application runs with read-only model artifacts and full prediction functionality.
