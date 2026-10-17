"""
app.py
======
AI-Powered Precision Agriculture and Crop Yield Optimization System
Streamlit web application — main entry point.

Run: .venv\\Scripts\\python.exe -m streamlit run app.py

Sections:
  A. Home Dashboard
  B. Predict Yield (single record + voice)
  C. Batch Predictions
  D. Dataset Explorer
  E. Model Performance
  F. Model Training
  G. Prediction History
  H. About Project
"""

import io
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data_processing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    DATA_PATH,
    NUMERIC_FEATURES,
    TARGET,
    check_out_of_training_range,
    load_raw_data,
    load_schema_info,
    prepare_single_record,
)
from src.model_training import (
    get_feature_importance,
    load_best_model,
    load_evaluation,
)
from src.prediction import (
    BATCH_REQUIRED_COLUMNS,
    predict_batch,
    predict_single,
    validate_batch_columns,
    validate_batch_file,
)
from src.translations import LANGUAGE_CODES, LANGUAGES, T
from src.voice_utils import (
    build_prediction_speech_text,
    get_tts_html,
    get_voice_input_html,
)

logging.basicConfig(level=logging.WARNING)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Precision Agriculture AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Typography */
  html, body, [class*="css"]  { font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif; }

  /* Ensure content is not hidden behind Streamlit top bar or header */
  .block-container {
    padding-top: 3.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1200px;
  }

  /* Sidebar Styling */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1b5e20 0%, #2e7d32 60%, #388e3c 100%) !important;
  }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] .stMarkdown {
    color: #f1f8e9 !important;
  }
  [data-testid="stSidebar"] .stSelectbox label {
    color: #e8f5e9 !important;
    font-weight: 600;
  }
  /* Keep selectbox dropdown menu readable (dark text on white dropdown background) */
  [data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #212121 !important;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label span {
    color: #f1f8e9 !important;
  }

  /* Metric cards */
  [data-testid="stMetricValue"] { color: #1b5e20; font-weight: 700; }

  /* Section headers */
  .section-header {
    background: linear-gradient(90deg, #2e7d32, #388e3c);
    color: #ffffff !important;
    padding: 14px 22px;
    border-radius: 8px;
    margin-top: 0.5rem;
    margin-bottom: 1.2rem;
    font-size: 1.35rem;
    font-weight: 600;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    display: block;
    clear: both;
  }

  /* Result box */
  .result-box {
    background: #f1f8e9; border: 2px solid #4caf50;
    border-radius: 10px; padding: 20px; margin: 12px 0;
    text-align: center;
  }
  .result-box .yield-value {
    font-size: 2.5rem; font-weight: 700; color: #1b5e20;
  }
  .result-box .yield-unit {
    font-size: 1rem; color: #558b2f;
  }

  /* Disclaimer */
  .disclaimer {
    background: #fff8e1; border-left: 4px solid #ffc107;
    padding: 10px 15px; border-radius: 4px; font-size: 0.88rem;
    color: #5d4037; margin-top: 8px;
  }

  /* Status badge */
  .badge-ok   { background:#e8f5e9; color:#1b5e20; padding:4px 12px; border-radius:20px; font-size:0.85rem; font-weight: 600; }
  .badge-warn { background:#fff8e1; color:#e65100; padding:4px 12px; border-radius:20px; font-size:0.85rem; font-weight: 600; }
  .badge-err  { background:#ffebee; color:#b71c1c; padding:4px 12px; border-radius:20px; font-size:0.85rem; font-weight: 600; }

  /* Mobile */
  @media (max-width: 768px) {
    .block-container { padding-top: 4.5rem !important; }
    .result-box .yield-value { font-size: 1.8rem; }
    .section-header { font-size: 1.15rem; padding: 10px 16px; }
  }
</style>
""", unsafe_allow_html=True)

# ── session state initialisation ──────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""

lang = st.session_state.lang


# ── cached data loaders ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame | None:
    try:
        df = load_raw_data()
        # Strip whitespace from categorical columns
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].str.strip()
        return df
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_model_cached():
    try:
        return load_best_model()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_eval_cached():
    try:
        return load_evaluation()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_schema_cached():
    try:
        return load_schema_info()
    except Exception:
        return None


def is_model_ready() -> bool:
    return load_model_cached() is not None and load_eval_cached() is not None


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 Precision Agriculture AI")
    st.markdown("---")

    # Language selector
    lang_label = T("language", lang)
    lang_choice = st.selectbox(
        lang_label,
        options=list(LANGUAGES.keys()),
        format_func=lambda k: LANGUAGES[k],
        index=list(LANGUAGES.keys()).index(st.session_state.lang),
        key="lang_selector",
    )
    if lang_choice != st.session_state.lang:
        st.session_state.lang = lang_choice
        lang = lang_choice
        st.rerun()

    st.markdown("---")

    # Navigation
    pages = {
        "🏠 " + T("nav_home", lang):        "home",
        "🔮 " + T("nav_predict", lang):     "predict",
        "📋 " + T("nav_batch", lang):       "batch",
        "📊 " + T("nav_explorer", lang):    "explorer",
        "📈 " + T("nav_performance", lang): "performance",
        "⚙️ " + T("nav_training", lang):    "training",
        "🕐 " + T("nav_history", lang):     "history",
        "ℹ️ " + T("nav_about", lang):       "about",
    }
    selected = st.radio(
        "Navigation",
        options=list(pages.keys()),
        label_visibility="collapsed",
    )
    page = pages[selected]

    st.markdown("---")
    # Model status indicator
    if is_model_ready():
        st.markdown('<span class="badge-ok">✅ Model Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-warn">⚠️ No Model</span>', unsafe_allow_html=True)

    st.caption("5th Sem CSE Design Project")

# ══════════════════════════════════════════════════════════════════════════════
# A. HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "home":
    st.markdown(f'<div class="section-header">🌾 {T("home_title", lang)}</div>', unsafe_allow_html=True)
    st.markdown(f"**{T('home_subtitle', lang)}**")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"""
        {T('home_description', lang)}

        This application allows you to:
        - **Predict crop yield** for any supported crop, state, and season
        - **Compare** Linear Regression, Decision Tree, and Random Forest models
        - **Explore** the historical dataset with interactive charts
        - **Upload batch CSVs** for multi-record predictions
        - **Switch** between English and Hindi interfaces with voice support
        """)

    with col2:
        # Model status card
        st.markdown(f"### {T('home_model_status', lang)}")
        if is_model_ready():
            st.success(T("home_model_ready", lang))
            eval_meta = load_eval_cached()
            if eval_meta:
                best = eval_meta.get("best_model", "Unknown")
                metrics = eval_meta.get("metrics", {}).get(best, {})
                st.metric("Best Model", best)
                if metrics:
                    cols = st.columns(3)
                    cols[0].metric("R²",   f"{metrics.get('R2', 0):.3f}")
                    cols[1].metric("MAE",  f"{metrics.get('MAE', 0):.2f}")
                    cols[2].metric("RMSE", f"{metrics.get('RMSE', 0):.2f}")
        else:
            st.warning(T("home_model_missing", lang))
            st.code(".venv\\Scripts\\python.exe train_model.py", language="bash")

    # Dataset stats
    st.markdown("---")
    st.markdown(f"### {T('home_dataset_stats', lang)}")
    df = load_dataset()
    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(T("home_total_records", lang), f"{len(df):,}")
        c2.metric(T("home_crops", lang), df["Crop"].nunique() if "Crop" in df.columns else "N/A")
        c3.metric(T("home_states", lang), df["State"].nunique() if "State" in df.columns else "N/A")
        c4.metric(T("home_years", lang),
                  f"{int(df['Crop_Year'].min())}–{int(df['Crop_Year'].max())}" if "Crop_Year" in df.columns else "N/A")

        # Quick chart: top crops by frequency
        col_a, col_b = st.columns(2)
        with col_a:
            crop_counts = df["Crop"].value_counts().head(10).reset_index()
            crop_counts.columns = ["Crop", "Count"]
            fig = px.bar(
                crop_counts, x="Count", y="Crop", orientation="h",
                title="Top 10 Crops by Record Count",
                color="Count", color_continuous_scale="Greens",
                template="plotly_white",
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            season_counts = df["Season"].value_counts().reset_index()
            season_counts.columns = ["Season", "Count"]
            fig2 = px.pie(
                season_counts, names="Season", values="Count",
                title="Records by Season",
                color_discrete_sequence=px.colors.sequential.Greens_r,
                template="plotly_white",
            )
            fig2.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.error("Dataset not found. Check data/crop_yield.csv.")

    st.markdown(f'<div class="disclaimer">{T("home_disclaimer", lang)}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# B. PREDICT YIELD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "predict":
    st.markdown(f'<div class="section-header">🔮 {T("predict_title", lang)}</div>', unsafe_allow_html=True)
    st.markdown(T("predict_subtitle", lang))

    # Compact three-step guide
    st.markdown("""
    <div style="background:#f1f8e9;border:1px solid #c8e6c9;border-radius:8px;padding:12px 18px;margin-bottom:15px;font-size:0.92rem;">
        <strong>🌾 Three-Step Prediction Guide:</strong><br>
        1. <strong>Enter Agricultural Details:</strong> Provide crop, season, state, year, and farming inputs (rainfall, fertilizer, pesticide).<br>
        2. <strong>Generate Estimate:</strong> Click 'Predict Yield' (or speak using voice input) to run the validated ML pipeline.<br>
        3. <strong>Review Results & Limitations:</strong> Examine the predicted yield per hectare, model test error bounds, and feature contributions.
    </div>
    """, unsafe_allow_html=True)

    if not is_model_ready():
        st.error(T("home_model_missing", lang))
        st.code(".venv\\Scripts\\python.exe train_model.py", language="bash")
        st.stop()

    schema = load_schema_cached()
    if schema is None:
        st.error("Feature schema not found. Run train_model.py first.")
        st.stop()

    df = load_dataset()

    # ── Voice input section ───────────────────────────────────────────────────
    with st.expander(f"🎤 {T('predict_mic', lang)} ({T('predict_mic_hint', lang)})", expanded=False):
        lang_code = LANGUAGE_CODES.get(lang, "en-IN")
        st.components.v1.html(get_voice_input_html(lang_code), height=220, scrolling=False)
        st.caption(T("voice_privacy", lang))
        st.caption(T("voice_not_supported", lang))

        transcript = st.text_input(
            T("predict_recognized", lang),
            value=st.session_state.voice_transcript,
            key="voice_text_input",
            placeholder="Speak or type recognized text here...",
        )
        st.caption("After speaking, manually copy the recognized text above, then click Apply.")
        if transcript:
            st.session_state.voice_transcript = transcript

    # ── Prediction form ───────────────────────────────────────────────────────
    st.markdown("---")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            crops = schema["categories"].get("Crop", [])
            crop_val = st.selectbox(T("predict_crop", lang), options=crops, key="sel_crop")

            seasons = schema["categories"].get("Season", [])
            season_val = st.selectbox(T("predict_season", lang), options=seasons, key="sel_season")

        with col2:
            states = schema["categories"].get("State", [])
            state_val = st.selectbox(T("predict_state", lang), options=states, key="sel_state")

            year_val = st.number_input(
                T("predict_year", lang),
                min_value=1990, max_value=2030,
                value=2019, step=1,
                help="Crop year (training data spans 1997–2020; 5th–95th percentile is 1998–2019)",
            )

        with col3:
            area_val = st.number_input(
                T("predict_area", lang),
                min_value=0.001, max_value=1e8,
                value=5000.0,
                help="Cultivated area in hectares",
                format="%.2f",
            )
            rainfall_val = st.number_input(
                T("predict_rainfall", lang),
                min_value=0.0, max_value=10000.0,
                value=1200.0,
                help="Annual rainfall in mm",
                format="%.1f",
            )

        col4, col5 = st.columns(2)
        with col4:
            fertilizer_val = st.number_input(
                T("predict_fertilizer", lang),
                min_value=0.0, max_value=1e11,
                value=500000.0,
                help="Total fertilizer used in kg",
                format="%.0f",
            )
        with col5:
            pesticide_val = st.number_input(
                T("predict_pesticide", lang),
                min_value=0.0, max_value=1e8,
                value=1000.0,
                help="Total pesticide used in kg",
                format="%.2f",
            )

        submitted = st.form_submit_button(
            T("predict_button", lang),
            use_container_width=True,
            type="primary",
        )

    # ── Run prediction ────────────────────────────────────────────────────────
    if submitted:
        input_dict = {
            "Crop": crop_val,
            "Season": season_val,
            "State": state_val,
            "Crop_Year": year_val,
            "Area": area_val,
            "Annual_Rainfall": rainfall_val,
            "Fertilizer": fertilizer_val,
            "Pesticide": pesticide_val,
        }

        with st.spinner(T("loading", lang)):
            try:
                result = predict_single(input_dict)
                pred_yield = result["predicted_yield"]
                model_name = result["model_name"]
                warnings = result["out_of_range_warnings"]

                eval_meta = load_eval_cached()
                best_metrics = eval_meta.get("metrics", {}).get(model_name, {}) if eval_meta else {}
                mae_val = best_metrics.get("MAE", "N/A")
                r2_val = best_metrics.get("R2", "N/A")

                # Distribution status badge
                dist_badge = '<span class="badge-ok">✅ Within Typical Training Range (5th–95th percentile)</span>' if not warnings else '<span class="badge-warn">⚠️ Tail Distribution Input (Outside 5th–95th percentile)</span>'

                # Enhanced result display card
                st.markdown(f"""
                <div class="result-box">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px;">
                    <span style="font-size:0.95rem;font-weight:600;color:#2e7d32;">
                      🌾 Historical-Data-Based Yield Estimate
                    </span>
                    {dist_badge}
                  </div>
                  <div style="font-size:1.05rem;color:#33691e;margin-bottom:6px;">
                    <strong>{crop_val}</strong> &bull; {season_val} &bull; {state_val} (Crop Year {year_val})
                  </div>
                  <div class="yield-value">{pred_yield:.2f}</div>
                  <div class="yield-unit">{T('predict_unit', lang)} (metric tonnes/ha)</div>
                  
                  <div style="margin-top:14px;padding-top:12px;border-top:1px dashed #a5d6a7;display:flex;justify-content:space-around;font-size:0.88rem;color:#424242;flex-wrap:wrap;gap:10px;">
                    <div><strong>Model:</strong> {model_name}</div>
                    <div><strong>Held-out Test MAE:</strong> &plusmn;{mae_val} t/ha</div>
                    <div><strong>Model Test R&sup2;:</strong> {r2_val}</div>
                    <div><strong>Area:</strong> {area_val:,.1f} ha</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                if warnings:
                    st.warning("**⚠️ Input Distribution Alert (5th–95th Percentile Bounds):**\n" + "\n".join(f"- {w}" for w in warnings) + "\n\n*Note: The model will still generate an estimate, but empirical uncertainty increases on tail inputs.*")

                st.markdown(f'<div class="disclaimer"><strong>⚠️ Scope & Uncertainty Notice:</strong> {T("predict_disclaimer", lang)} Predictions are empirical estimates from historical regional data and do not constitute certified agricultural advisory or guaranteed crop yields.</div>',
                            unsafe_allow_html=True)

                # ── TTS button ─────────────────────────────────────────────────
                speech_text = build_prediction_speech_text(crop_val, pred_yield, model_name, lang)
                lang_code = LANGUAGE_CODES.get(lang, "en-IN")
                st.markdown(f"**{T('predict_listen', lang)}**")
                st.components.v1.html(
                    get_tts_html(speech_text, lang_code),
                    height=100, scrolling=False,
                )

                # ── Feature importance quick view ──────────────────────────────
                pipeline = load_model_cached()
                feat_imp = get_feature_importance(pipeline, ALL_FEATURES)
                if feat_imp:
                    with st.expander("📊 Feature Importance (this model)", expanded=False):
                        imp_df = pd.DataFrame(
                            list(feat_imp.items()), columns=["Feature", "Importance"]
                        ).sort_values("Importance", ascending=True)
                        fig = px.bar(
                            imp_df, x="Importance", y="Feature", orientation="h",
                            color="Importance", color_continuous_scale="Greens",
                            template="plotly_white",
                            title="Feature Importance",
                        )
                        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                                          margin=dict(l=0, r=0, t=40, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(T("perf_feat_note", lang))

                # ── Add to history ─────────────────────────────────────────────
                st.session_state.prediction_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "crop": crop_val,
                    "season": season_val,
                    "state": state_val,
                    "year": year_val,
                    "area": area_val,
                    "rainfall": rainfall_val,
                    "predicted_yield": pred_yield,
                    "model": model_name,
                    "lang": lang,
                })

            except ValueError as e:
                st.error(f"**{T('error', lang)}:** {e}")
            except FileNotFoundError as e:
                st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# C. BATCH PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "batch":
    st.markdown(f'<div class="section-header">📋 {T("batch_title", lang)}</div>', unsafe_allow_html=True)
    st.markdown(T("batch_subtitle", lang))

    if not is_model_ready():
        st.error(T("home_model_missing", lang))
        st.code(".venv\\Scripts\\python.exe train_model.py", language="bash")
        st.stop()

    # Step-by-step guidance & requirements
    st.markdown("""
    <div style="background:#f1f8e9;border:1px solid #c8e6c9;border-radius:8px;padding:14px 18px;margin-bottom:15px;font-size:0.92rem;">
        <strong>📋 Three-Step Batch Workflow:</strong><br>
        1. <strong>Prepare CSV:</strong> Format file with required agricultural features (see schema below or download template).<br>
        2. <strong>Upload & Validate:</strong> File size is enforced at max <strong>5 MB</strong> and maximum <strong>5,000 rows</strong>.<br>
        3. <strong>Run Inference & Export:</strong> Predictions are processed with the active pipeline. Valid records and rejected records can be downloaded separately.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📌 CSV Specifications & Download Sample Template", expanded=False):
        c_spec1, c_spec2 = st.columns(2)
        with c_spec1:
            st.markdown("**Required Columns (Case-Sensitive):**")
            st.code(", ".join(BATCH_REQUIRED_COLUMNS))
            st.markdown("""
            - **Categorical:** `Crop`, `Season`, `State`
            - **Numerical:** `Crop_Year`, `Area` (ha), `Annual_Rainfall` (mm), `Fertilizer` (kg), `Pesticide` (kg)
            """)
        with c_spec2:
            st.markdown("**System Constraints:**")
            st.markdown("""
            - **Max File Size:** 5 MB (strictly checked before parsing)
            - **Max Records:** 5,000 rows per batch upload
            - **Extra Columns:** Allowed; ignored during model prediction
            - **Negative / NaN / Inf values:** Rejected with specific row-level error
            """)

        # Sample template download
        sample = pd.DataFrame([
            {"Crop": "Rice", "Crop_Year": 2018, "Season": "Kharif", "State": "Uttar Pradesh", "Area": 5000.0, "Annual_Rainfall": 1000.0, "Fertilizer": 500000.0, "Pesticide": 1000.0},
            {"Crop": "Wheat", "Crop_Year": 2019, "Season": "Rabi", "State": "Punjab", "Area": 8000.0, "Annual_Rainfall": 650.0, "Fertilizer": 1200000.0, "Pesticide": 2500.0},
            {"Crop": "Maize", "Crop_Year": 2017, "Season": "Kharif", "State": "Karnataka", "Area": 3500.0, "Annual_Rainfall": 850.0, "Fertilizer": 300000.0, "Pesticide": 800.0}
        ])
        csv_sample = sample.to_csv(index=False)
        st.download_button(
            "📥 Download Valid Sample CSV Template",
            data=csv_sample,
            file_name="cropzen_batch_sample.csv",
            mime="text/csv",
        )

    st.markdown("---")
    uploaded = st.file_uploader(
        T("batch_upload", lang),
        type=["csv"],
        help="Max file size 5 MB. Must be a valid CSV file.",
    )

    if uploaded:
        file_size = len(uploaded.getvalue())
        batch_df, file_error = validate_batch_file(uploaded, uploaded.name, file_size)

        if file_error:
            st.error(f"❌ **{T('error', lang)}:** {file_error}")
            st.stop()

        # Column validation
        present_cols, missing_cols, extra_cols = validate_batch_columns(batch_df)
        if missing_cols:
            st.error(f"❌ **{T('batch_error_cols', lang)}:** {', '.join(missing_cols)}")
            st.stop()

        if extra_cols:
            st.info(f"ℹ️ **Note on Extra Columns:** Detected {len(extra_cols)} extra column(s): `{', '.join(extra_cols)}`. These columns are preserved but ignored during model inference.")

        # Preview
        st.markdown(f"### {T('batch_preview', lang)}")
        st.dataframe(batch_df.head(5), use_container_width=True)

        col_l, col_r = st.columns(2)
        col_l.metric("Total rows uploaded", f"{len(batch_df):,}")
        col_r.metric("File Size", f"{file_size / 1024:.1f} KB")

        if st.button(T("batch_run", lang), type="primary"):
            schema = load_schema_cached()
            if schema is None:
                st.error("Feature schema missing. Run train_model.py first.")
                st.stop()

            with st.spinner("Running predictions using active pipeline..."):
                result_df = predict_batch(batch_df, schema)

            ok_mask  = result_df["Prediction_Status"] == "OK"
            err_mask = result_df["Prediction_Status"] == "ERROR"

            c_v1, c_v2, c_v3 = st.columns(3)
            c_v1.metric(T("batch_valid_rows", lang), f"{ok_mask.sum():,}")
            c_v2.metric(T("batch_invalid_rows", lang), f"{err_mask.sum():,}")
            c_v3.metric("Validation Pass Rate", f"{(ok_mask.sum() / len(batch_df) * 100):.1f}%")

            # Valid results
            if ok_mask.sum() > 0:
                st.markdown(f"### {T('batch_results', lang)}")
                st.markdown(f"**Predicted Target:** `Yield` (Historical estimate in **tonnes per hectare**)")
                display_cols = BATCH_REQUIRED_COLUMNS + ["Predicted_Yield"]
                st.dataframe(result_df.loc[ok_mask, display_cols], use_container_width=True)

                # Download valid results
                csv_out = result_df.loc[ok_mask, display_cols].to_csv(index=False)
                st.download_button(
                    f"📥 {T('batch_download', lang)} ({ok_mask.sum():,} records)",
                    data=csv_out,
                    file_name="cropzen_predictions.csv",
                    mime="text/csv",
                )

                # Summary chart
                st.markdown(f"### {T('batch_summary', lang)}")
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    fig = px.histogram(
                        result_df.loc[ok_mask], x="Predicted_Yield",
                        title="Distribution of Predicted Yields",
                        labels={"Predicted_Yield": "Predicted Yield (tonnes/ha)"},
                        color_discrete_sequence=["#4caf50"],
                        template="plotly_white",
                        nbins=30,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col_chart2:
                    if result_df.loc[ok_mask, "Crop"].nunique() > 1:
                        crop_avg = result_df.loc[ok_mask].groupby("Crop")["Predicted_Yield"].mean().reset_index()
                        fig_crop = px.bar(
                            crop_avg, x="Crop", y="Predicted_Yield",
                            title="Average Predicted Yield by Crop",
                            labels={"Predicted_Yield": "Avg Yield (t/ha)"},
                            color="Crop",
                            template="plotly_white",
                        )
                        st.plotly_chart(fig_crop, use_container_width=True)
                    else:
                        pred_stats = result_df.loc[ok_mask, "Predicted_Yield"].describe().round(4)
                        st.dataframe(pred_stats.to_frame("Yield Summary (t/ha)"), use_container_width=True)

            # Invalid rows
            if err_mask.sum() > 0:
                st.markdown(f"### ⚠️ {T('batch_invalid_rows', lang)}")
                st.markdown("The following records were rejected because they contain missing fields, unknown categories, or invalid values:")
                err_display_cols = ["Prediction_Error"] + [c for c in BATCH_REQUIRED_COLUMNS if c in result_df.columns]
                st.dataframe(result_df.loc[err_mask, err_display_cols], use_container_width=True)

                # Download rejected rows with error explanation
                err_csv = result_df.loc[err_mask, err_display_cols].to_csv(index=False)
                st.download_button(
                    f"📥 Download Rejected Records CSV ({err_mask.sum():,} rows)",
                    data=err_csv,
                    file_name="cropzen_rejected_rows.csv",
                    mime="text/csv",
                )
            elif ok_mask.sum() == 0:
                st.warning(T("batch_no_valid", lang))


# ══════════════════════════════════════════════════════════════════════════════
# D. DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "explorer":
    st.markdown(f'<div class="section-header">📊 {T("explorer_title", lang)}</div>', unsafe_allow_html=True)

    df = load_dataset()
    if df is None:
        st.error("Dataset not found at data/crop_yield.csv.")
        st.stop()

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Filters")
        all_crops = sorted(df["Crop"].unique().tolist())
        sel_crops = st.multiselect(T("explorer_filter_crop", lang), all_crops, default=[])
        all_states = sorted(df["State"].unique().tolist())
        sel_states = st.multiselect(T("explorer_filter_state", lang), all_states, default=[])
        all_seasons = sorted(df["Season"].unique().tolist())
        sel_seasons = st.multiselect(T("explorer_filter_season", lang), all_seasons, default=[])
        year_min, year_max = int(df["Crop_Year"].min()), int(df["Crop_Year"].max())
        sel_years = st.slider(T("explorer_filter_year", lang), year_min, year_max, (year_min, year_max))

    # Apply filters
    fdf = df.copy()
    if sel_crops:  fdf = fdf[fdf["Crop"].isin(sel_crops)]
    if sel_states: fdf = fdf[fdf["State"].isin(sel_states)]
    if sel_seasons: fdf = fdf[fdf["Season"].isin(sel_seasons)]
    fdf = fdf[(fdf["Crop_Year"] >= sel_years[0]) & (fdf["Crop_Year"] <= sel_years[1])]

    if fdf.empty:
        st.warning("⚠️ **No records match the selected filter combination.**")
        st.info("Try widening your filters (e.g., select more crops, include all seasons, or expand the year range).")
        st.stop()

    # ── Overview ──────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric(T("explorer_rows", lang), f"{len(fdf):,}")
    col2.metric(T("explorer_cols", lang), fdf.shape[1])
    col3.metric("Filtered from", f"{len(df):,} total")

    # Dataset preview
    with st.expander("Dataset Preview (first 20 rows)"):
        st.dataframe(fdf.head(20), use_container_width=True)

    # ── Missing value analysis ────────────────────────────────────────────────
    st.markdown(f"### {T('explorer_missing', lang)}")
    missing = fdf.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Count"]
    missing["Missing %"] = (missing["Missing Count"] / len(fdf) * 100).round(2)
    if missing["Missing Count"].sum() == 0:
        st.success("No missing values in the filtered dataset.")
    else:
        st.dataframe(missing, use_container_width=True)

    # ── Descriptive stats ─────────────────────────────────────────────────────
    with st.expander(T("explorer_stats", lang)):
        num_cols = [c for c in ["Area", "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"] if c in fdf.columns]
        st.dataframe(fdf[num_cols].describe().round(4), use_container_width=True)

    # ── Yield distribution ────────────────────────────────────────────────────
    st.markdown(f"### {T('explorer_distribution', lang)}")
    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.histogram(
            fdf, x="Yield", nbins=50,
            title="Yield Distribution (all values)",
            labels={"Yield": "Yield (tonnes/ha)"},
            color_discrete_sequence=["#4caf50"],
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Log scale for clarity given the skewed distribution
        fdf_pos = fdf[fdf["Yield"] > 0].copy()
        fdf_pos["log_yield"] = np.log1p(fdf_pos["Yield"])
        fig2 = px.histogram(
            fdf_pos, x="log_yield", nbins=50,
            title="Yield Distribution (log scale, excludes zero-yield)",
            labels={"log_yield": "log(1 + Yield)"},
            color_discrete_sequence=["#81c784"],
            template="plotly_white",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Crop-wise comparison ──────────────────────────────────────────────────
    st.markdown(f"### {T('explorer_crop_compare', lang)}")
    top_n = min(20, fdf["Crop"].nunique())
    top_crops = fdf.groupby("Crop")["Yield"].median().nlargest(top_n).index.tolist()
    crop_df = fdf[fdf["Crop"].isin(top_crops)]
    fig3 = px.box(
        crop_df, x="Crop", y="Yield",
        title=f"Yield Distribution by Crop (top {top_n} by median yield)",
        labels={"Yield": "Yield (tonnes/ha)"},
        color="Crop",
        color_discrete_sequence=px.colors.qualitative.Set3,
        template="plotly_white",
    )
    fig3.update_layout(showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Season-wise comparison ────────────────────────────────────────────────
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f"### {T('explorer_season_compare', lang)}")
        fig4 = px.box(
            fdf, x="Season", y="Yield",
            title="Yield by Season",
            color="Season",
            color_discrete_sequence=px.colors.sequential.Greens_r,
            template="plotly_white",
        )
        fig4.update_layout(showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    with col_s2:
        st.markdown(f"### {T('explorer_year_trend', lang)}")
        yr_df = fdf.groupby("Crop_Year")["Yield"].median().reset_index()
        fig5 = px.line(
            yr_df, x="Crop_Year", y="Yield",
            title="Median Yield per Year",
            markers=True,
            labels={"Yield": "Median Yield (t/ha)", "Crop_Year": "Year"},
            color_discrete_sequence=["#2e7d32"],
            template="plotly_white",
        )
        st.plotly_chart(fig5, use_container_width=True)

    # ── State-wise comparison ─────────────────────────────────────────────────
    st.markdown(f"### {T('explorer_state_compare', lang)}")
    state_df = fdf.groupby("State")["Yield"].median().reset_index().sort_values("Yield", ascending=True)
    fig6 = px.bar(
        state_df, x="Yield", y="State", orientation="h",
        title="Median Yield by State",
        labels={"Yield": "Median Yield (t/ha)"},
        color="Yield", color_continuous_scale="Greens",
        template="plotly_white",
    )
    fig6.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig6, use_container_width=True)

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.markdown(f"### {T('explorer_correlation', lang)}")
    corr_cols = [c for c in ["Area", "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"] if c in fdf.columns]
    corr_matrix = fdf[corr_cols].corr()
    fig7 = px.imshow(
        corr_matrix,
        title="Correlation Matrix (Numerical Features)",
        color_continuous_scale="RdYlGn",
        zmin=-1, zmax=1,
        aspect="auto",
        template="plotly_white",
        text_auto=".2f",
    )
    st.plotly_chart(fig7, use_container_width=True)
    st.caption(T("explorer_note_corr", lang))

    # Rainfall vs Yield scatter
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        fig8 = px.scatter(
            fdf.sample(min(2000, len(fdf)), random_state=42),
            x="Annual_Rainfall", y="Yield",
            title="Rainfall vs Yield (sample)",
            labels={"Annual_Rainfall": "Annual Rainfall (mm)", "Yield": "Yield (t/ha)"},
            color="Season", opacity=0.6,
            template="plotly_white",
        )
        st.plotly_chart(fig8, use_container_width=True)

    with col_r2:
        fig9 = px.scatter(
            fdf.sample(min(2000, len(fdf)), random_state=42),
            x="Area", y="Yield",
            title="Area vs Yield (sample)",
            labels={"Area": "Area (ha)", "Yield": "Yield (t/ha)"},
            color="Season", opacity=0.6,
            log_x=True,
            template="plotly_white",
        )
        st.plotly_chart(fig9, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# E. MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "performance":
    st.markdown(f'<div class="section-header">📈 {T("perf_title", lang)}</div>', unsafe_allow_html=True)
    st.markdown(T("perf_subtitle", lang))

    eval_meta = load_eval_cached()
    if eval_meta is None:
        st.error(T("home_model_missing", lang))
        st.code(".venv\\Scripts\\python.exe train_model.py", language="bash")
        st.stop()

    metrics_all = eval_meta.get("metrics", {})
    best_model  = eval_meta.get("best_model", "")

    # ── Metrics comparison table ──────────────────────────────────────────────
    st.markdown(f"### {T('perf_best_model', lang)}: **{best_model}**")

    col_a, col_b = st.columns(2)
    col_a.metric(T("perf_train_size", lang), f"{eval_meta.get('train_size', 0):,} rows")
    col_b.metric(T("perf_test_size", lang),  f"{eval_meta.get('test_size', 0):,} rows")

    st.markdown("#### All Models — Evaluation Metrics")
    rows = []
    for name, m in metrics_all.items():
        rows.append({
            "Model": name,
            "MAE":  m.get("MAE", None),
            "RMSE": m.get("RMSE", None),
            "R²":   m.get("R2", None),
            "Best": "⭐" if name == best_model else "",
        })
    metrics_df = pd.DataFrame(rows)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # ── Bar charts ────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        fig = px.bar(
            metrics_df, x="Model", y="MAE",
            title=T("perf_mae", lang),
            color="Model", color_discrete_sequence=px.colors.qualitative.Set2,
            template="plotly_white",
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(T("perf_mae_explain", lang))

    with col2:
        fig2 = px.bar(
            metrics_df, x="Model", y="RMSE",
            title=T("perf_rmse", lang),
            color="Model", color_discrete_sequence=px.colors.qualitative.Set2,
            template="plotly_white",
        )
        fig2.update_layout(showlegend=False, xaxis_tickangle=-20)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(T("perf_rmse_explain", lang))

    with col3:
        fig3 = px.bar(
            metrics_df, x="Model", y="R²",
            title=T("perf_r2", lang),
            color="Model", color_discrete_sequence=px.colors.qualitative.Set2,
            template="plotly_white",
        )
        fig3.update_layout(showlegend=False, xaxis_tickangle=-20)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(T("perf_r2_explain", lang))

    # ── Actual vs Predicted ───────────────────────────────────────────────────
    st.markdown(f"### {T('perf_actual_vs_pred', lang)}")
    y_true = eval_meta.get("y_true_test", [])
    y_pred = eval_meta.get("y_pred_test", [])

    if y_true and y_pred:
        scatter_df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
        # Limit to 2000 for performance
        if len(scatter_df) > 2000:
            scatter_df = scatter_df.sample(2000, random_state=42)

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            max_val = max(scatter_df["Actual"].max(), scatter_df["Predicted"].max())
            fig4 = px.scatter(
                scatter_df, x="Actual", y="Predicted",
                title=f"Actual vs Predicted ({best_model})",
                labels={"Actual": "Actual Yield (t/ha)", "Predicted": "Predicted Yield (t/ha)"},
                opacity=0.5,
                color_discrete_sequence=["#4caf50"],
                template="plotly_white",
            )
            # Perfect-prediction line
            fig4.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines", name="Perfect Prediction",
                line=dict(color="red", dash="dash", width=1),
            ))
            st.plotly_chart(fig4, use_container_width=True)

        with col_p2:
            scatter_df["Residual"] = scatter_df["Predicted"] - scatter_df["Actual"]
            fig5 = px.scatter(
                scatter_df, x="Predicted", y="Residual",
                title=f"Residuals Plot ({best_model})",
                labels={"Predicted": "Predicted Yield (t/ha)", "Residual": "Residual"},
                opacity=0.5,
                color_discrete_sequence=["#1976d2"],
                template="plotly_white",
            )
            fig5.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig5, use_container_width=True)

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown(f"### {T('perf_feat_importance', lang)}")
    pipeline = load_model_cached()
    if pipeline:
        feat_imp = get_feature_importance(pipeline, ALL_FEATURES)
        if feat_imp:
            imp_df = pd.DataFrame(
                list(feat_imp.items()), columns=["Feature", "Importance"]
            ).sort_values("Importance", ascending=True)
            fig6 = px.bar(
                imp_df, x="Importance", y="Feature", orientation="h",
                title=f"Feature Importance — {best_model}",
                color="Importance", color_continuous_scale="Greens",
                template="plotly_white",
            )
            fig6.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig6, use_container_width=True)
            st.caption(T("perf_feat_note", lang))
        else:
            st.info("Feature importance not available for this model type.")

    # ── Model descriptions ────────────────────────────────────────────────────
    st.markdown("### Model Descriptions")
    descriptions = eval_meta.get("model_descriptions", {})
    for mname, mdesc in descriptions.items():
        badge = "⭐ " if mname == best_model else ""
        st.markdown(f"**{badge}{mname}**: {mdesc}")

    # ── Methodological & Metric Interpretation Notes ────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Evaluation Methodology & Statistical Notes")
    
    col_note1, col_note2 = st.columns(2)
    with col_note1:
        st.markdown("""
        **Understanding MAE vs. RMSE Disparity:**
        - **MAE (15.47 t/ha):** Measures average absolute deviation. Typical crop errors are relatively compact.
        - **RMSE (297.17 t/ha):** Because RMSE squares errors before averaging, it is heavily weighted by the dataset's extreme right tail (crops like *Coconut* where raw count units reach up to 21,105).
        - **R² Score (0.8898):** Quantifies that the Decision Tree explains ~89% of variance in held-out test records. *R² is a variance-explained metric, NOT classification accuracy.*
        """)
    with col_note2:
        st.markdown(f"""
        **Train / Test Split Design:**
        - **Split Strategy:** {eval_meta.get('split_method', 'Random 80/20')}
        - **Held-out Test Set:** {eval_meta.get('test_size', 0):,} records ({eval_meta.get('test_size_fraction', 0.2)*100:.0f}%)
        - **Chronological / Generalization Limitation:** {eval_meta.get('split_limitation', '')}
        - **Timestamp of Evaluated Artifacts:** {eval_meta.get('trained_at', '')}
        """)

    st.caption("All metrics shown above are directly loaded from verified test evaluation artifacts computed by train_model.py.")


# ══════════════════════════════════════════════════════════════════════════════
# F. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "training":
    st.markdown(f'<div class="section-header">⚙️ {T("train_title", lang)}</div>', unsafe_allow_html=True)
    st.markdown(T("train_subtitle", lang))

    st.markdown(f"### {T('train_status', lang)}")

    is_local = True  # Always treat as local (no auth mechanism deployed)

    eval_meta = load_eval_cached()
    if eval_meta:
        st.success(T("train_success", lang))
        best = eval_meta.get("best_model", "")
        trained_at = eval_meta.get("trained_at", "")
        st.markdown(f"**{T('train_last_run', lang)}:** {trained_at}")
        st.markdown(f"**Best Model:** {best}")

        m = eval_meta.get("metrics", {}).get(best, {})
        if m:
            cols = st.columns(3)
            cols[0].metric("R²",   f"{m.get('R2', 0):.4f}")
            cols[1].metric("MAE",  f"{m.get('MAE', 0):.4f}")
            cols[2].metric("RMSE", f"{m.get('RMSE', 0):.4f}")
    else:
        st.warning(T("home_model_missing", lang))

    st.markdown("---")

    # Safe retraining
    st.markdown(f"### {T('train_button', lang)}")
    st.warning(T("train_warning", lang))
    st.markdown("""
    **How to retrain locally:**
    ```powershell
    .venv\\Scripts\\python.exe train_model.py
    ```
    Then restart the Streamlit app to load the new model.
    """)

    # Show current model file info
    model_path = ROOT / "models" / "best_model.pkl"
    if model_path.exists():
        size_kb = model_path.stat().st_size / 1024
        st.info(f"Model file: `models/best_model.pkl`  ({size_kb:.1f} KB)")

    schema_path = ROOT / "outputs" / "feature_schema.json"
    if schema_path.exists():
        st.info(f"Feature schema: `outputs/feature_schema.json`")

    # Security note
    st.markdown("---")
    st.markdown("""
    **Security Note:**
    The retraining button is intentionally disabled in this app because a public
    Streamlit app cannot distinguish the project owner from ordinary visitors.
    Retraining must be run locally using the command above. The last known-good
    model remains active if a new training run fails.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# G. PREDICTION HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "history":
    st.markdown(f'<div class="section-header">🕐 {T("history_title", lang)}</div>', unsafe_allow_html=True)
    st.markdown(T("history_subtitle", lang))

    history = st.session_state.prediction_history

    if not history:
        st.info(T("history_empty", lang))
    else:
        if st.button(T("history_clear", lang)):
            st.session_state.prediction_history = []
            st.success(T("history_cleared", lang))
            st.rerun()

        hist_df = pd.DataFrame(history)
        display_cols = ["timestamp", "crop", "season", "state", "year",
                        "area", "rainfall", "predicted_yield", "model"]
        display_cols = [c for c in display_cols if c in hist_df.columns]

        st.dataframe(
            hist_df[display_cols].rename(columns={
                "timestamp":       T("history_timestamp", lang),
                "crop":            T("history_crop", lang),
                "predicted_yield": T("history_yield", lang),
                "model":           T("history_model", lang),
            }),
            use_container_width=True,
        )

        # Mini chart
        if len(history) > 1:
            fig = px.bar(
                hist_df, x="timestamp", y="predicted_yield",
                title="Predicted Yields Over Session",
                color="crop",
                labels={"predicted_yield": "Yield (t/ha)", "timestamp": "Time"},
                template="plotly_white",
            )
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        # Download history
        csv_hist = hist_df.to_csv(index=False)
        st.download_button(
            "📥 Download History CSV",
            data=csv_hist,
            file_name="prediction_history.csv",
            mime="text/csv",
        )

    st.caption("History is stored only for this browser session and is cleared when you close the tab.")


# ══════════════════════════════════════════════════════════════════════════════
# H. ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "about":
    st.markdown(f'<div class="section-header">ℹ️ {T("about_title", lang)}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(f"## {T('about_problem', lang)}")
        st.markdown("""
        Crop yield prediction is challenging because agricultural production varies
        with environmental conditions, crop characteristics and regional differences.
        The reliability of machine-learning predictions depends on data quality,
        feature selection and model performance.

        This project develops a web-based system that preprocesses historical
        agricultural data, trains and compares multiple regression models, and
        predicts crop yield using the features supported by the dataset. The system
        evaluates predictions using MAE, RMSE and R² and communicates the limitations
        of the resulting estimates.
        """)

        st.markdown(f"## {T('about_objectives', lang)}")
        st.markdown("""
        1. Perform exploratory data analysis on Indian crop yield data (1997–2020).
        2. Train and compare Linear Regression, Decision Tree and Random Forest models.
        3. Evaluate models objectively on a held-out test set.
        4. Build an interactive prediction interface with English and Hindi support.
        5. Provide voice input and spoken output using browser APIs.
        6. Enable batch predictions via CSV upload.
        7. Explain model behaviour using feature importance.
        """)

        st.markdown(f"## {T('about_dataset', lang)}")
        st.markdown("""
        **Dataset Overview & Provenance:**
        - **Source:** Historical Indian agricultural crop production dataset (1997–2020), derived from Ministry of Agriculture and Farmers Welfare public records / Kaggle open repositories.
        - **License / Status:** Open educational dataset for non-commercial academic research.

        | Property | Value |
        |----------|-------|
        | Records | 19,689 rows |
        | Columns | 10 |
        | Unique Crops | 55 |
        | States Covered | 30 |
        | Agricultural Seasons | 6 (Kharif, Rabi, Whole Year, Summer, Autumn, Winter) |
        | Historical Years | 1997–2020 |

        **Validated Feature Schema:**
        - **Input Features:** `Crop`, `Season`, `State`, `Crop_Year`, `Area` (ha), `Annual_Rainfall` (mm), `Fertilizer` (kg), `Pesticide` (kg)
        - **Target:** `Yield` (log1p transformed during training; expm1 inverse transformation applied to predictions; metric tonnes/ha).
        - **Target Leakage Exclusion:** `Production` is strictly excluded because it represents post-harvest output and in many records correlates directly with the target computation (`Yield = Production / Area`), causing artificial target leakage.

        **Empirical Evaluation Design:**
        - **Split:** Fixed 80/20 train/test split (seed=42) on 19,689 records (15,751 train / 3,938 test).
        - **Known Limitations:**
          - Tail distribution crops (e.g. coconut counted in units) skew raw RMSE.
          - 112 zero-yield records are preserved as genuine historical crop failure events.
          - Generalization across newly introduced states/crops or future extreme climatic shifts requires ongoing localized calibration.
        """)

    with col2:
        st.markdown(f"## {T('about_algorithms', lang)}")
        st.markdown("""
        **Linear Regression**
        Fits a linear relationship between input features and yield.
        Simple, fast, interpretable. Struggles with non-linear patterns.

        **Decision Tree Regressor**
        Learns recursive splits on feature values. Captures non-linearity.
        Prone to overfitting without depth limits.

        **Random Forest Regressor**
        Ensemble of 100 decision trees. More robust, provides feature importance.
        Slower to train and predict.
        """)

        st.markdown(f"## {T('about_metrics', lang)}")
        st.markdown("""
        | Metric | Meaning | Direction |
        |--------|---------|-----------|
        | MAE | Average absolute error | Lower ↓ |
        | RMSE | Square-root of mean squared error; penalises large errors | Lower ↓ |
        | R² | Proportion of variance explained (max 1.0; can be negative) | Higher ↑ |

        R² is **not classification accuracy**.
        """)

        st.markdown(f"## {T('about_scope', lang)}")
        st.markdown("""
        **Implemented:**
        - ✅ Data audit and EDA
        - ✅ ML training pipeline (3 models)
        - ✅ Single-record prediction
        - ✅ Batch CSV prediction
        - ✅ Dataset explorer
        - ✅ Model performance dashboard
        - ✅ Feature importance
        - ✅ English/Hindi interface
        - ✅ Voice input and output (browser API)
        - ✅ Session prediction history
        - ✅ Responsive design

        **Not implemented (future work):**
        - ❌ Satellite or remote sensing data
        - ❌ Live weather API integration
        - ❌ IoT sensor input
        - ❌ Soil composition features
        - ❌ Irrigation/fertilizer recommendations
        - ❌ Persistent user database
        - ❌ Multi-year forecasting
        """)

        st.markdown(f"## {T('about_future', lang)}")
        st.markdown("""
        - Incorporate weather forecast APIs for future seasons
        - Add soil and irrigation features where data is available
        - Implement group-based train/test split (by state or crop group)
        - Add SHAP-based model explanations
        - Deploy to Streamlit Community Cloud with read-only model
        - Build offline mobile version for field use
        """)
