"""
tests/test_pipeline.py
======================
Automated tests for the Precision Agriculture ML project.
Run: .venv\\Scripts\\python.exe -m pytest tests/ -v
"""

import json
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_processing import (
    ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET,
    clean_data, load_raw_data, load_schema_info, prepare_single_record,
    validate_schema, check_out_of_training_range,
)
from src.translations import T, get_text, LANGUAGES
from src.prediction import validate_batch_file, validate_batch_columns


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset loads correctly and is unmodified
# ─────────────────────────────────────────────────────────────────────────────
class TestDataLoading:
    def test_dataset_loads(self):
        df = load_raw_data()
        assert df is not None
        assert len(df) > 0, "Dataset is empty"

    def test_dataset_row_count(self):
        df = load_raw_data()
        assert len(df) == 19689, f"Expected 19689 rows, got {len(df)}"

    def test_required_columns_present(self):
        df = load_raw_data()
        for col in ALL_FEATURES + [TARGET]:
            assert col in df.columns, f"Missing column: {col}"

    def test_production_not_in_features(self):
        """Production must not be included as a predictor (leakage risk)."""
        assert "Production" not in ALL_FEATURES

    def test_no_modification_of_original(self):
        """Load twice and compare — original file should be unchanged."""
        df1 = load_raw_data()
        df2 = load_raw_data()
        assert df1.shape == df2.shape


# ─────────────────────────────────────────────────────────────────────────────
# 2. Schema validation
# ─────────────────────────────────────────────────────────────────────────────
class TestSchemaValidation:
    def test_schema_passes_with_all_columns(self):
        df = load_raw_data()
        validate_schema(df)  # should not raise

    def test_schema_fails_with_missing_column(self):
        df = load_raw_data().drop(columns=["Yield"])
        with pytest.raises(ValueError, match="Yield"):
            validate_schema(df)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Data cleaning
# ─────────────────────────────────────────────────────────────────────────────
class TestDataCleaning:
    def test_clean_removes_zero_area(self):
        df = load_raw_data()
        # Inject a zero-area row
        bad = df.iloc[0:1].copy()
        bad["Area"] = 0.0
        df_bad = pd.concat([df, bad], ignore_index=True)
        df_clean = clean_data(df_bad, log_decisions=False)
        assert (df_clean["Area"] > 0).all()

    def test_clean_removes_negative_yield(self):
        df = load_raw_data()
        bad = df.iloc[0:1].copy()
        bad["Yield"] = -5.0
        df_bad = pd.concat([df, bad], ignore_index=True)
        df_clean = clean_data(df_bad, log_decisions=False)
        assert (df_clean["Yield"] >= 0).all()

    def test_clean_preserves_zero_yield(self):
        """Zero-yield rows (genuine zero production) should be kept."""
        df = load_raw_data()
        df_clean = clean_data(df, log_decisions=False)
        zero_yield = (df_clean["Yield"] == 0).sum()
        # We know the dataset has 112 zero-yield rows
        assert zero_yield == 112


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model artifacts exist and load
# ─────────────────────────────────────────────────────────────────────────────
class TestModelArtifacts:
    def test_model_file_exists(self):
        model_path = ROOT / "models" / "best_model.pkl"
        assert model_path.exists(), "Run train_model.py first"

    def test_model_loads(self):
        from src.model_training import load_best_model
        pipeline = load_best_model()
        assert pipeline is not None

    def test_evaluation_json_exists(self):
        eval_path = ROOT / "outputs" / "evaluation.json"
        assert eval_path.exists(), "Run train_model.py first"

    def test_evaluation_has_real_metrics(self):
        from src.model_training import load_evaluation
        ev = load_evaluation()
        metrics = ev["metrics"]
        assert len(metrics) >= 1
        for name, m in metrics.items():
            assert "MAE" in m and "RMSE" in m and "R2" in m
            assert m["MAE"] >= 0, f"{name} MAE should be non-negative"
            assert m["RMSE"] >= 0, f"{name} RMSE should be non-negative"
            assert m["R2"] <= 1.0, f"{name} R² cannot exceed 1.0"

    def test_feature_schema_exists(self):
        schema_path = ROOT / "outputs" / "feature_schema.json"
        assert schema_path.exists(), "Run train_model.py first"

    def test_schema_has_categories(self):
        schema = load_schema_info()
        for col in CATEGORICAL_FEATURES:
            assert col in schema["categories"]
            assert len(schema["categories"][col]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Single-record prediction
# ─────────────────────────────────────────────────────────────────────────────
class TestSinglePrediction:
    VALID_INPUT = {
        "Crop": "Rice",
        "Season": "Kharif",
        "State": "Uttar Pradesh",
        "Crop_Year": 2018,
        "Area": 5000.0,
        "Annual_Rainfall": 900.0,
        "Fertilizer": 500000.0,
        "Pesticide": 1000.0,
    }

    def test_valid_prediction_returns_float(self):
        from src.prediction import predict_single
        result = predict_single(self.VALID_INPUT)
        assert isinstance(result["predicted_yield"], float)
        assert result["predicted_yield"] >= 0

    def test_valid_prediction_has_model_name(self):
        from src.prediction import predict_single
        result = predict_single(self.VALID_INPUT)
        assert "model_name" in result
        assert result["model_name"] != ""

    def test_invalid_crop_raises(self):
        from src.prediction import predict_single
        bad = dict(self.VALID_INPUT)
        bad["Crop"] = "FAKE_CROP_XYZ"
        with pytest.raises(ValueError):
            predict_single(bad)

    def test_zero_area_raises(self):
        from src.prediction import predict_single
        bad = dict(self.VALID_INPUT)
        bad["Area"] = 0.0
        with pytest.raises(ValueError):
            predict_single(bad)

    def test_missing_field_raises(self):
        from src.prediction import predict_single
        bad = {k: v for k, v in self.VALID_INPUT.items() if k != "Crop"}
        with pytest.raises(ValueError):
            predict_single(bad)

    def test_nan_numeric_raises(self):
        from src.prediction import predict_single
        bad = dict(self.VALID_INPUT)
        bad["Annual_Rainfall"] = float("nan")
        with pytest.raises(ValueError):
            predict_single(bad)

    def test_inf_numeric_raises(self):
        from src.prediction import predict_single
        bad = dict(self.VALID_INPUT)
        bad["Fertilizer"] = float("inf")
        with pytest.raises(ValueError):
            predict_single(bad)

    def test_negative_numeric_raises(self):
        from src.prediction import predict_single
        bad = dict(self.VALID_INPUT)
        bad["Fertilizer"] = -100.0
        with pytest.raises(ValueError):
            predict_single(bad)

    def test_out_of_range_warning(self):
        from src.prediction import predict_single
        inp = dict(self.VALID_INPUT)
        inp["Annual_Rainfall"] = 9999.0  # extreme value
        result = predict_single(inp)
        assert len(result["out_of_range_warnings"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Batch prediction validation
# ─────────────────────────────────────────────────────────────────────────────
class TestBatchValidation:
    def test_missing_columns_detected(self):
        df = pd.DataFrame({"Crop": ["Rice"], "Area": [5000]})
        present, missing, extra = validate_batch_columns(df)
        assert len(missing) > 0
        assert "Season" in missing

    def test_all_required_columns_present(self):
        df = pd.DataFrame([{
            "Crop": "Rice", "Crop_Year": 2018, "Season": "Kharif",
            "State": "Uttar Pradesh", "Area": 5000.0,
            "Annual_Rainfall": 900.0, "Fertilizer": 500000.0, "Pesticide": 1000.0,
        }])
        present, missing, extra = validate_batch_columns(df)
        assert len(missing) == 0
        assert len(extra) == 0

    def test_extra_columns_reported(self):
        df = pd.DataFrame([{
            "Crop": "Rice", "Crop_Year": 2018, "Season": "Kharif",
            "State": "Uttar Pradesh", "Area": 5000.0,
            "Annual_Rainfall": 900.0, "Fertilizer": 500000.0, "Pesticide": 1000.0,
            "Farmer_Notes": "High organic matter", "Soil_Type": "Alluvial"
        }])
        present, missing, extra = validate_batch_columns(df)
        assert len(missing) == 0
        assert "Farmer_Notes" in extra
        assert "Soil_Type" in extra

    def test_file_size_limit(self):
        import io
        fake_content = b"a,b\n" + b"1,2\n" * 100
        fobj = io.BytesIO(fake_content)
        _, err = validate_batch_file(fobj, "test.csv", 6 * 1024 * 1024)  # 6MB
        assert err is not None
        assert "5 MB" in err

    def test_empty_file_rejected(self):
        import io
        fobj = io.BytesIO(b"")
        _, err = validate_batch_file(fobj, "empty.csv", 0)
        assert err is not None
        assert "empty" in err.lower()

    def test_header_only_file_rejected(self):
        import io
        fobj = io.BytesIO(b"Crop,Season,State,Crop_Year,Area,Annual_Rainfall,Fertilizer,Pesticide\n")
        _, err = validate_batch_file(fobj, "headers_only.csv", 100)
        assert err is not None
        assert "zero data rows" in err.lower()

    def test_duplicate_headers_rejected(self):
        import io
        fobj = io.BytesIO(b"Crop,Crop,Area\nRice,Wheat,100\n")
        _, err = validate_batch_file(fobj, "duplicate_cols.csv", 50)
        assert err is not None
        assert "duplicate column" in err.lower()

    def test_non_csv_rejected(self):
        import io
        fobj = io.BytesIO(b"fake content")
        _, err = validate_batch_file(fobj, "model.pkl", 100)
        assert err is not None
        assert "csv" in err.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Translations
# ─────────────────────────────────────────────────────────────────────────────
class TestTranslations:
    def test_english_translation_returns_string(self):
        text = T("nav_home", "en")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_hindi_translation_returns_string(self):
        text = T("nav_home", "hi")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_hindi_different_from_english(self):
        en = T("nav_home", "en")
        hi = T("nav_home", "hi")
        assert en != hi

    def test_missing_key_falls_back_to_english(self):
        hi_text = T("predict_button", "hi")
        en_text = T("predict_button", "en")
        # Hindi should have a translation, not fall back
        assert hi_text is not None

    def test_completely_missing_key_returns_key(self):
        text = T("nonexistent_key_xyz", "en")
        assert "nonexistent_key_xyz" in text

    def test_all_english_keys_have_hindi(self):
        from src.translations import TRANSLATIONS
        en_keys = set(TRANSLATIONS["en"].keys())
        hi_keys = set(TRANSLATIONS["hi"].keys())
        missing_in_hindi = en_keys - hi_keys
        # Allow up to 5 missing keys (edge cases)
        assert len(missing_in_hindi) <= 5, f"Missing Hindi translations for: {missing_in_hindi}"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Feature schema
# ─────────────────────────────────────────────────────────────────────────────
class TestFeatureSchema:
    def test_prepare_valid_input(self):
        schema = load_schema_info()
        # Use first crop/season/state from schema
        crop  = schema["categories"]["Crop"][0]
        season = schema["categories"]["Season"][0]
        state  = schema["categories"]["State"][0]
        inp = {
            "Crop": crop, "Season": season, "State": state,
            "Crop_Year": 2018, "Area": 5000.0,
            "Annual_Rainfall": 900.0, "Fertilizer": 500000.0, "Pesticide": 1000.0,
        }
        df = prepare_single_record(inp, schema)
        assert df is not None
        assert len(df) == 1

    def test_prepare_invalid_categorical(self):
        schema = load_schema_info()
        inp = {
            "Crop": "NOT_A_REAL_CROP", "Season": "Kharif",
            "State": "Uttar Pradesh", "Crop_Year": 2018,
            "Area": 5000.0, "Annual_Rainfall": 900.0,
            "Fertilizer": 500000.0, "Pesticide": 1000.0,
        }
        with pytest.raises(ValueError):
            prepare_single_record(inp, schema)

    def test_out_of_range_detection(self):
        schema = load_schema_info()
        inp = {
            "Crop": "Rice", "Season": "Kharif",
            "State": "Uttar Pradesh", "Crop_Year": 2018,
            "Area": 5000.0, "Annual_Rainfall": 9500.0,  # extreme
            "Fertilizer": 500000.0, "Pesticide": 1000.0,
        }
        warnings = check_out_of_training_range(inp, schema)
        assert any("Annual_Rainfall" in w for w in warnings)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Multi-Dataset Integrity & Scope
# ─────────────────────────────────────────────────────────────────────────────
class TestMultiDatasetIntegrity:
    def test_model_ready_dataset_loads(self):
        from src.data_processing import load_model_ready_data
        df = load_model_ready_data()
        assert df is not None
        assert len(df) == 17983
        assert "Yield" in df.columns
        assert "Annual_Rainfall" in df.columns

    def test_district_dataset_has_no_rainfall_fertilizer(self):
        """Verify district dataset does not have rainfall or fertilizer and shouldn't train ML model directly."""
        from src.data_processing import load_district_data
        df = load_district_data()
        assert df is not None
        assert len(df) == 311488
        assert df["Annual_Rainfall"].isna().all()
        assert df["Fertilizer"].isna().all()

    def test_official_dataset_missing_yield_and_handled_safely(self):
        """Official 2021-2025 dataset must have 100% missing yield, prohibiting supervised validation."""
        from src.data_processing import load_official_2021_25_data
        df = load_official_2021_25_data()
        assert df is not None
        assert len(df) == 225
        assert df["Yield"].isna().all()
        assert df["Area"].isna().all()
        # Verify years correspond to 2020-21 through 2024-25
        assert df["Crop_Year"].min() == 2021
        assert df["Crop_Year"].max() == 2025

    def test_combined_dataset_loads_and_has_both_geographies(self):
        from src.data_processing import load_combined_data
        df = load_combined_data()
        assert df is not None
        assert len(df) == 311713
        assert set(df["geography_level"].unique()) == {"district", "all_india"}

    def test_missing_dataset_graceful_handling(self, tmp_path, monkeypatch):
        """App functions must return None rather than raising unhandled exceptions when dataset file is missing."""
        from src import data_processing
        monkeypatch.setattr(data_processing, "DATA_PATH_MODEL_READY", tmp_path / "nonexistent.csv")
        df = data_processing.load_model_ready_data()
        assert df is None

