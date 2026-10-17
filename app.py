"""
app.py
======
CropZen — AI-Powered Precision Agriculture and Crop Yield Optimization System
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
    page_title="CropZen — Precision Agriculture AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System CSS ─────────────────────────────────────────────────────────
# Palette:
#   Primary green   #1F7A4D   Dark green     #14532D
#   Light green bg  #F2F8F4   Accent yellow  #EAB308
#   Warning orange  #F59E0B   Error red      #DC2626
#   Main text       #1F2937   Muted text     #6B7280
#   Card bg         #FFFFFF   Page bg        #F7FAF8
st.markdown("""
<style>
/* ── Base typography ───────────────────────────────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
  color: #1F2937;
}

/* ── Page background ───────────────────────────────────────────────────────── */
.stApp {
  background-color: #F7FAF8 !important;
}

/* ── Block container ───────────────────────────────────────────────────────── */
.block-container {
  padding-top: 1.5rem !important;
  padding-bottom: 3rem !important;
  max-width: 1200px;
}

/* ── Sidebar redesign (light, clean) ───────────────────────────────────────── */
[data-testid="stSidebar"] {
  background-color: #F2F8F4 !important;
  border-right: 1px solid #D1E7DA !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown {
  color: #1F2937 !important;
}
[data-testid="stSidebar"] .stSelectbox label {
  color: #1F2937 !important;
  font-weight: 600;
  font-size: 0.85rem;
}
[data-testid="stSidebar"] [data-baseweb="select"] * {
  color: #1F2937 !important;
}
/* Navigation radio — style as pill cards */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  padding: 8px 12px !important;
  border-radius: 8px !important;
  margin: 2px 0 !important;
  display: block !important;
  transition: background 0.15s ease;
  cursor: pointer;
  font-size: 0.9rem;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
  background: #DCEFE6 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] + div label,
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
  background: #FFFFFF !important;
  border-left: 4px solid #1F7A4D !important;
  font-weight: 600 !important;
  color: #1F7A4D !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label span {
  color: #1F2937 !important;
}

/* ── App header / top bar ───────────────────────────────────────────────────── */
.cropzen-topbar {
  background: #FFFFFF;
  border-bottom: 1px solid #D1E7DA;
  padding: 10px 0 10px 0;
  margin-bottom: 1.5rem;
}
.cropzen-logo {
  font-size: 1.6rem;
  font-weight: 700;
  color: #1F7A4D;
  display: flex;
  align-items: center;
  gap: 8px;
}
.cropzen-tagline {
  font-size: 0.8rem;
  color: #6B7280;
  font-weight: 400;
  margin-top: 1px;
}

/* ── Page title block ───────────────────────────────────────────────────────── */
.page-title {
  font-size: 1.65rem;
  font-weight: 700;
  color: #1F2937;
  margin-bottom: 0.25rem;
  line-height: 1.3;
}
.page-subtitle {
  font-size: 0.95rem;
  color: #6B7280;
  margin-bottom: 1.25rem;
}
.page-breadcrumb {
  font-size: 0.78rem;
  color: #6B7280;
  margin-bottom: 0.5rem;
}

/* ── Section divider ────────────────────────────────────────────────────────── */
.section-title {
  font-size: 1.15rem;
  font-weight: 600;
  color: #1F2937;
  margin-top: 1.5rem;
  margin-bottom: 0.75rem;
  padding-bottom: 6px;
  border-bottom: 2px solid #D1E7DA;
}

/* ── White cards ────────────────────────────────────────────────────────────── */
.card {
  background: #FFFFFF;
  border: 1px solid #E5F0EA;
  border-radius: 12px;
  padding: 20px 24px;
  margin-bottom: 16px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.card-accent {
  border-left: 4px solid #1F7A4D;
}

/* ── Hero banner ────────────────────────────────────────────────────────────── */
.hero-banner {
  background: linear-gradient(135deg, #1F7A4D 0%, #14532D 100%);
  border-radius: 16px;
  padding: 36px 40px;
  margin-bottom: 24px;
  color: #FFFFFF;
}
.hero-banner h1 {
  font-size: 2rem;
  font-weight: 700;
  color: #FFFFFF !important;
  margin-bottom: 10px;
}
.hero-banner p {
  font-size: 1.05rem;
  color: #D1FAE5 !important;
  margin-bottom: 0;
}

/* ── Action cards (home) ────────────────────────────────────────────────────── */
.action-card {
  background: #FFFFFF;
  border: 1px solid #E5F0EA;
  border-radius: 12px;
  padding: 22px 20px;
  text-align: center;
  transition: box-shadow 0.2s ease, transform 0.15s ease;
  cursor: pointer;
  height: 100%;
}
.action-card:hover {
  box-shadow: 0 6px 18px rgba(31,122,77,0.15);
  transform: translateY(-2px);
}
.action-card .action-icon {
  font-size: 2.2rem;
  margin-bottom: 10px;
  display: block;
}
.action-card .action-title {
  font-size: 1rem;
  font-weight: 600;
  color: #1F2937;
  margin-bottom: 6px;
}
.action-card .action-desc {
  font-size: 0.83rem;
  color: #6B7280;
}

/* ── Stat cards ─────────────────────────────────────────────────────────────── */
.stat-card {
  background: #FFFFFF;
  border: 1px solid #E5F0EA;
  border-radius: 10px;
  padding: 16px 18px;
  text-align: center;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.stat-card .stat-value {
  font-size: 1.85rem;
  font-weight: 700;
  color: #1F7A4D;
  line-height: 1.2;
}
.stat-card .stat-label {
  font-size: 0.78rem;
  color: #6B7280;
  margin-top: 4px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

/* ── Step guide ─────────────────────────────────────────────────────────────── */
.steps-guide {
  background: #F2F8F4;
  border: 1px solid #C9E8D5;
  border-radius: 10px;
  padding: 14px 18px;
  margin-bottom: 18px;
  font-size: 0.9rem;
  color: #1F2937;
}
.steps-guide strong { color: #1F7A4D; }

/* ── How it works steps ─────────────────────────────────────────────────────── */
.how-step {
  display: flex;
  align-items: flex-start;
  gap: 14px;
  margin-bottom: 14px;
}
.how-step-num {
  background: #1F7A4D;
  color: #FFFFFF;
  font-weight: 700;
  font-size: 0.9rem;
  min-width: 30px;
  height: 30px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.how-step-body strong { color: #1F2937; font-size: 0.95rem; }
.how-step-body p { color: #6B7280; font-size: 0.83rem; margin: 2px 0 0 0; }

/* ── Result card ────────────────────────────────────────────────────────────── */
.result-card {
  background: #FFFFFF;
  border: 2px solid #1F7A4D;
  border-radius: 14px;
  padding: 28px 32px;
  margin: 18px 0;
  text-align: center;
  box-shadow: 0 4px 16px rgba(31,122,77,0.12);
}
.result-card .yield-number {
  font-size: 3rem;
  font-weight: 700;
  color: #1F7A4D;
  line-height: 1.1;
}
.result-card .yield-unit {
  font-size: 1rem;
  color: #6B7280;
  margin-top: 4px;
}
.result-card .result-context {
  font-size: 0.85rem;
  color: #6B7280;
  margin-top: 6px;
}
.result-card .result-meta {
  margin-top: 18px;
  padding-top: 14px;
  border-top: 1px dashed #C9E8D5;
  display: flex;
  justify-content: center;
  gap: 32px;
  flex-wrap: wrap;
  font-size: 0.83rem;
  color: #374151;
}
.result-card .result-meta span strong { color: #1F2937; }

/* ── Disclaimer / notice boxes ──────────────────────────────────────────────── */
.disclaimer {
  background: #FFFBEB;
  border-left: 4px solid #EAB308;
  padding: 10px 16px;
  border-radius: 6px;
  font-size: 0.85rem;
  color: #78350F;
  margin-top: 10px;
}
.info-notice {
  background: #EFF6FF;
  border-left: 4px solid #3B82F6;
  padding: 10px 16px;
  border-radius: 6px;
  font-size: 0.85rem;
  color: #1E3A5F;
  margin: 8px 0;
}
.limitation-alert {
  background: #FEF2F2;
  border: 1px solid #FECACA;
  border-left: 5px solid #DC2626;
  border-radius: 8px;
  padding: 14px 18px;
  font-size: 0.9rem;
  color: #7F1D1D;
  margin-bottom: 18px;
}
.security-note {
  background: #F2F8F4;
  border: 1px solid #C9E8D5;
  border-left: 5px solid #1F7A4D;
  border-radius: 8px;
  padding: 14px 18px;
  font-size: 0.9rem;
  color: #14532D;
  margin-bottom: 18px;
}

/* ── Badges ─────────────────────────────────────────────────────────────────── */
.badge-ok   { background:#DCFCE7; color:#14532D; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-warn { background:#FEF9C3; color:#713F12; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-err  { background:#FEE2E2; color:#7F1D1D; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-info { background:#DBEAFE; color:#1E3A5F; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }

/* ── Metrics (override Streamlit default green) ─────────────────────────────── */
[data-testid="stMetricValue"] { color: #1F7A4D !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #6B7280 !important; font-size: 0.8rem !important; }

/* ── Buttons ─────────────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
  background-color: #1F7A4D !important;
  border-color: #1F7A4D !important;
  color: #FFFFFF !important;
  font-weight: 600;
  border-radius: 8px;
  padding: 0.5rem 1.5rem;
}
.stButton > button[kind="primary"]:hover {
  background-color: #14532D !important;
  border-color: #14532D !important;
}
.stButton > button[kind="secondary"] {
  border-color: #D1E7DA !important;
  border-radius: 8px;
}

/* ── Tabs ────────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab"] {
  font-size: 0.9rem;
  font-weight: 500;
}
[data-testid="stTabs"] [aria-selected="true"] {
  color: #1F7A4D !important;
  border-bottom-color: #1F7A4D !important;
}

/* ── Mobile ──────────────────────────────────────────────────────────────────── */
@media (max-width: 768px) {
  .block-container { padding-top: 3rem !important; }
  .result-card .yield-number { font-size: 2.2rem; }
  .hero-banner { padding: 22px 20px; }
  .hero-banner h1 { font-size: 1.5rem; }
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


# ── Helper: page header ───────────────────────────────────────────────────────
def render_page_header(icon: str, title: str, subtitle: str = "", breadcrumb: str = "CropZen"):
    if breadcrumb:
        st.markdown(f'<div class="page-breadcrumb">🏠 {breadcrumb}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">{icon} {title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo block
    st.markdown("""
    <div style="padding: 12px 4px 8px 4px;">
      <div style="font-size:1.35rem;font-weight:700;color:#1F7A4D;display:flex;align-items:center;gap:8px;">
        🌾 CropZen
      </div>
      <div style="font-size:0.72rem;color:#6B7280;margin-top:2px;">
        AI Precision Agriculture
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border:none;border-top:1px solid #D1E7DA;margin:8px 0 12px 0;" />', unsafe_allow_html=True)

    # Language selector
    lang_choice = st.selectbox(
        "🌐 " + T("language", lang),
        options=list(LANGUAGES.keys()),
        format_func=lambda k: LANGUAGES[k],
        index=list(LANGUAGES.keys()).index(st.session_state.lang),
        key="lang_selector",
    )
    if lang_choice != st.session_state.lang:
        st.session_state.lang = lang_choice
        lang = lang_choice
        st.rerun()

    st.markdown('<hr style="border:none;border-top:1px solid #D1E7DA;margin:12px 0;" />', unsafe_allow_html=True)

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

    st.markdown('<hr style="border:none;border-top:1px solid #D1E7DA;margin:12px 0 8px 0;" />', unsafe_allow_html=True)

    # Model status
    if is_model_ready():
        eval_meta = load_eval_cached()
        best = eval_meta.get("best_model", "Unknown") if eval_meta else "Unknown"
        st.markdown(f'<span class="badge-ok">✅ Model Ready · {best}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-warn">⚠️ Model Not Found</span>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.72rem;color:#6B7280;margin-top:12px;">5th Sem CSE Design Project</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# A. HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "home":
    # Hero banner
    st.markdown(f"""
    <div class="hero-banner">
      <h1>🌾 {T('home_title', lang)}</h1>
      <p>{T('home_description', lang)}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── 3 Primary action cards ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">Get Started</div>', unsafe_allow_html=True)
    act1, act2, act3 = st.columns(3)

    with act1:
        st.markdown("""
        <div class="action-card">
          <span class="action-icon">🔮</span>
          <div class="action-title">Predict Crop Yield</div>
          <div class="action-desc">Enter agricultural parameters and get an instant ML-powered yield estimate with error bounds.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Predict →", key="btn_home_predict", use_container_width=True):
            st.session_state["_nav_override"] = "predict"
            st.rerun()

    with act2:
        st.markdown("""
        <div class="action-card">
          <span class="action-icon">📊</span>
          <div class="action-title">Explore Dataset</div>
          <div class="action-desc">Visualise 19,689 historical crop records — filter by crop, state, season, and year range.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Explorer →", key="btn_home_explore", use_container_width=True):
            st.session_state["_nav_override"] = "explorer"
            st.rerun()

    with act3:
        st.markdown("""
        <div class="action-card">
          <span class="action-icon">📈</span>
          <div class="action-title">Compare Models</div>
          <div class="action-desc">Benchmark Linear Regression, Decision Tree, and Random Forest on MAE, RMSE, and R².</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Performance →", key="btn_home_perf", use_container_width=True):
            st.session_state["_nav_override"] = "performance"
            st.rerun()

    # Handle nav overrides from action buttons
    if "_nav_override" in st.session_state:
        _target = st.session_state.pop("_nav_override")
        page = _target

    # ── 4 Dataset stat cards ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">Dataset at a Glance</div>', unsafe_allow_html=True)
    df = load_dataset()
    if df is not None:
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-value">{len(df):,}</div>
              <div class="stat-label">{T('home_total_records', lang)}</div>
            </div>
            """, unsafe_allow_html=True)
        with s2:
            crops_count = df["Crop"].nunique() if "Crop" in df.columns else "—"
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-value">{crops_count}</div>
              <div class="stat-label">{T('home_crops', lang)}</div>
            </div>
            """, unsafe_allow_html=True)
        with s3:
            states_count = df["State"].nunique() if "State" in df.columns else "—"
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-value">{states_count}</div>
              <div class="stat-label">{T('home_states', lang)}</div>
            </div>
            """, unsafe_allow_html=True)
        with s4:
            year_range = (
                f"{int(df['Crop_Year'].min())}–{int(df['Crop_Year'].max())}"
                if "Crop_Year" in df.columns else "—"
            )
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-value" style="font-size:1.3rem;">{year_range}</div>
              <div class="stat-label">{T('home_years', lang)}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Model status inline card ───────────────────────────────────────────
        st.markdown("")
        if is_model_ready():
            eval_meta = load_eval_cached()
            best = eval_meta.get("best_model", "Unknown") if eval_meta else "Unknown"
            metrics = eval_meta.get("metrics", {}).get(best, {}) if eval_meta else {}
            r2  = metrics.get("R2",   0)
            mae = metrics.get("MAE",  0)
            rmse= metrics.get("RMSE", 0)
            st.markdown(f"""
            <div class="card card-accent" style="margin-top:8px;">
              <div style="font-size:0.85rem;color:#6B7280;margin-bottom:6px;">Active Model</div>
              <div style="font-size:1.1rem;font-weight:700;color:#1F7A4D;margin-bottom:8px;">⭐ {best}</div>
              <div style="display:flex;gap:32px;flex-wrap:wrap;">
                <div><div style="font-size:1.4rem;font-weight:700;color:#1F7A4D;">{r2:.4f}</div><div style="font-size:0.75rem;color:#6B7280;">R² (Test)</div></div>
                <div><div style="font-size:1.4rem;font-weight:700;color:#1F2937;">{mae:.2f}</div><div style="font-size:0.75rem;color:#6B7280;">MAE (t/ha)</div></div>
                <div><div style="font-size:1.4rem;font-weight:700;color:#1F2937;">{rmse:.2f}</div><div style="font-size:0.75rem;color:#6B7280;">RMSE (t/ha)</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="border-left:4px solid #F59E0B;">
              <div style="color:#78350F;font-weight:600;">⚠️ No trained model found</div>
              <div style="font-size:0.85rem;color:#6B7280;margin-top:4px;">Run <code>.venv\\Scripts\\python.exe train_model.py</code> to train the model first.</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Quick charts ───────────────────────────────────────────────────────
        st.markdown('<div class="section-title">Quick Data Insights</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            crop_counts = df["Crop"].value_counts().head(10).reset_index()
            crop_counts.columns = ["Crop", "Count"]
            fig = px.bar(
                crop_counts, x="Count", y="Crop", orientation="h",
                title="Top 10 Crops by Number of Records",
                color="Count", color_continuous_scale=["#C9E8D5", "#1F7A4D"],
                template="plotly_white",
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              margin=dict(l=0, r=0, t=40, b=0),
                              font=dict(color="#1F2937"),
                              plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            season_counts = df["Season"].value_counts().reset_index()
            season_counts.columns = ["Season", "Count"]
            fig2 = px.pie(
                season_counts, names="Season", values="Count",
                title="Distribution of Records by Season",
                color_discrete_sequence=["#1F7A4D", "#2D9D63", "#4DB87E", "#80CEA0", "#B3E2C7", "#E0F4EC"],
                template="plotly_white",
            )
            fig2.update_layout(margin=dict(l=0, r=0, t=40, b=0),
                               font=dict(color="#1F2937"),
                               paper_bgcolor="#FFFFFF")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.error("Dataset not found. Check data/crop_yield.csv.")

    # ── How it works ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
    hw1, hw2, hw3, hw4 = st.columns(4)
    steps = [
        ("1", "📥 Provide Inputs", "Select crop, state, season, year, area, rainfall, fertilizer and pesticide amounts."),
        ("2", "⚙️ ML Pipeline", "Inputs pass through a validated preprocessing pipeline and the trained regression model."),
        ("3", "🎯 Yield Estimate", "Predicted yield (tonnes/ha) is returned with model error bounds (±MAE) for context."),
        ("4", "📤 Export & Analyse", "Save single predictions to session history or upload a CSV for batch analysis."),
    ]
    for col, (num, title, desc) in zip([hw1, hw2, hw3, hw4], steps):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:18px;">
              <div style="background:#1F7A4D;color:#fff;font-weight:700;width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin:0 auto 10px auto;font-size:1rem;">{num}</div>
              <div style="font-weight:600;color:#1F2937;font-size:0.9rem;margin-bottom:6px;">{title}</div>
              <div style="color:#6B7280;font-size:0.78rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f'<div class="disclaimer">⚠️ {T("home_disclaimer", lang)}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# B. PREDICT YIELD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "predict":
    render_page_header(
        "🔮", T("predict_title", lang),
        subtitle=T("predict_subtitle", lang),
        breadcrumb="CropZen › Predict Yield",
    )

    if not is_model_ready():
        st.error(T("home_model_missing", lang))
        st.code(".venv\\Scripts\\python.exe train_model.py", language="bash")
        st.stop()

    schema = load_schema_cached()
    if schema is None:
        st.error("Feature schema not found. Run train_model.py first.")
        st.stop()

    df = load_dataset()

    # Step guide
    st.markdown("""
    <div class="steps-guide">
      <strong>🌾 Three-Step Prediction Guide</strong><br>
      <strong>Step 1 — Crop & Location:</strong> Choose which crop, which state, and which agricultural season.<br>
      <strong>Step 2 — Time & Land:</strong> Specify the crop year and cultivated area in hectares.<br>
      <strong>Step 3 — Environment & Inputs:</strong> Enter rainfall, fertilizer, and pesticide quantities.
    </div>
    """, unsafe_allow_html=True)

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

    st.markdown("---")

    # ── Prediction form ───────────────────────────────────────────────────────
    with st.form("prediction_form"):
        # Step 1: Crop & Location
        st.markdown('<div class="section-title">Step 1 — Crop & Location</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            crops = schema["categories"].get("Crop", [])
            crop_val = st.selectbox(
                T("predict_crop", lang),
                options=crops, key="sel_crop",
                help="Select the crop variety from the 55 supported crops.",
            )
        with col2:
            states = schema["categories"].get("State", [])
            state_val = st.selectbox(
                T("predict_state", lang),
                options=states, key="sel_state",
                help="Select the Indian state where the crop was/will be cultivated.",
            )

        seasons = schema["categories"].get("Season", [])
        season_val = st.selectbox(
            T("predict_season", lang),
            options=seasons, key="sel_season",
            help="Agricultural season: Kharif (June–Oct), Rabi (Oct–Mar), Whole Year, Summer, Autumn, or Winter.",
        )

        # Step 2: Time & Land
        st.markdown('<div class="section-title">Step 2 — Time & Land</div>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            year_val = st.number_input(
                T("predict_year", lang),
                min_value=1990, max_value=2030,
                value=2019, step=1,
                help="Crop year (training data spans 1997–2020; 5th–95th percentile: 1998–2019).",
            )
        with col4:
            area_val = st.number_input(
                T("predict_area", lang) + " (ha)",
                min_value=0.001, max_value=1e8,
                value=5000.0,
                help="Cultivated area in hectares. Typical range: 100 – 500,000 ha.",
                format="%.2f",
            )

        # Step 3: Environment & Inputs
        st.markdown('<div class="section-title">Step 3 — Environment & Agricultural Inputs</div>', unsafe_allow_html=True)
        col5, col6, col7 = st.columns(3)
        with col5:
            rainfall_val = st.number_input(
                T("predict_rainfall", lang) + " (mm)",
                min_value=0.0, max_value=10000.0,
                value=1200.0,
                help="Annual rainfall in mm. Typical range: 200 – 3,000 mm.",
                format="%.1f",
            )
        with col6:
            fertilizer_val = st.number_input(
                T("predict_fertilizer", lang) + " (kg)",
                min_value=0.0, max_value=1e11,
                value=500000.0,
                help="Total fertilizer used in kg. Typical range: 1,000 – 10,000,000 kg.",
                format="%.0f",
            )
        with col7:
            pesticide_val = st.number_input(
                T("predict_pesticide", lang) + " (kg)",
                min_value=0.0, max_value=1e8,
                value=1000.0,
                help="Total pesticide used in kg. Typical range: 10 – 50,000 kg.",
                format="%.2f",
            )

        submitted = st.form_submit_button(
            f"🔮 {T('predict_button', lang)}",
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
                r2_val  = best_metrics.get("R2",  "N/A")

                dist_badge = (
                    '<span class="badge-ok">✅ Within Typical Range (5th–95th percentile)</span>'
                    if not warnings else
                    '<span class="badge-warn">⚠️ Tail Distribution Input</span>'
                )
                mae_display = f"{mae_val:.2f}" if isinstance(mae_val, (int, float)) else mae_val
                r2_display  = f"{r2_val:.4f}"  if isinstance(r2_val,  (int, float)) else r2_val

                st.markdown(f"""
                <div class="result-card">
                  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:12px;">
                    <span style="font-size:0.88rem;color:#6B7280;">Historical-Data-Based Yield Estimate</span>
                    {dist_badge}
                  </div>
                  <div style="font-size:0.95rem;color:#374151;margin-bottom:10px;">
                    <strong>{crop_val}</strong> &bull; {season_val} &bull; {state_val} &bull; Year {year_val}
                  </div>
                  <div class="yield-number">{pred_yield:.2f}</div>
                  <div class="yield-unit">{T('predict_unit', lang)} &mdash; metric tonnes per hectare</div>
                  <div class="result-context">
                    Estimated total production for {area_val:,.0f} ha:
                    <strong>{pred_yield * area_val:,.0f} tonnes</strong>
                  </div>
                  <div class="result-meta">
                    <span><strong>Model:</strong> {model_name}</span>
                    <span><strong>Test MAE:</strong> &plusmn;{mae_display} t/ha</span>
                    <span><strong>Test R&sup2;:</strong> {r2_display}</span>
                    <span><strong>Area:</strong> {area_val:,.1f} ha</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                if warnings:
                    st.warning(
                        "**⚠️ Input Distribution Alert (outside 5th–95th training percentiles):**\n"
                        + "\n".join(f"- {w}" for w in warnings)
                        + "\n\n*The model will still generate an estimate, but empirical uncertainty increases on tail inputs.*"
                    )

                st.markdown(
                    f'<div class="disclaimer"><strong>⚠️ Scope & Uncertainty Notice:</strong> {T("predict_disclaimer", lang)} '
                    f'Predictions are empirical estimates from historical regional data and do not constitute certified agricultural advisory or guaranteed crop yields.</div>',
                    unsafe_allow_html=True,
                )

                # ── TTS button ─────────────────────────────────────────────────
                speech_text = build_prediction_speech_text(crop_val, pred_yield, model_name, lang)
                lang_code = LANGUAGE_CODES.get(lang, "en-IN")
                st.markdown(f"**{T('predict_listen', lang)}**")
                st.components.v1.html(
                    get_tts_html(speech_text, lang_code),
                    height=100, scrolling=False,
                )

                # ── Feature importance ──────────────────────────────────────────
                pipeline = load_model_cached()
                feat_imp = get_feature_importance(pipeline, ALL_FEATURES)
                if feat_imp:
                    with st.expander("📊 Feature Importance — What drove this prediction?", expanded=True):
                        imp_df = pd.DataFrame(
                            list(feat_imp.items()), columns=["Feature", "Importance"]
                        ).sort_values("Importance", ascending=True)
                        fig = px.bar(
                            imp_df, x="Importance", y="Feature", orientation="h",
                            color="Importance",
                            color_continuous_scale=["#C9E8D5", "#1F7A4D"],
                            template="plotly_white",
                            title=f"Feature Importance — {model_name}",
                        )
                        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                                          margin=dict(l=0, r=0, t=40, b=0),
                                          font=dict(color="#1F2937"),
                                          plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
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
    render_page_header(
        "📋", T("batch_title", lang),
        subtitle=T("batch_subtitle", lang),
        breadcrumb="CropZen › Batch Predictions",
    )

    if not is_model_ready():
        st.error(T("home_model_missing", lang))
        st.code(".venv\\Scripts\\python.exe train_model.py", language="bash")
        st.stop()

    st.markdown("""
    <div class="steps-guide">
      <strong>📋 Three-Step Batch Workflow</strong><br>
      <strong>Step 1 — Prepare CSV:</strong> Format your file with the required agricultural features (see schema below, or download a sample template).<br>
      <strong>Step 2 — Upload & Validate:</strong> File size is enforced at max <strong>5 MB</strong> and maximum <strong>5,000 rows</strong>.<br>
      <strong>Step 3 — Run & Export:</strong> Valid rows receive predictions; rejected rows include error explanations. Both can be downloaded separately.
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
            - **Max File Size:** 5 MB (enforced before parsing)
            - **Max Records:** 5,000 rows per batch upload
            - **Extra Columns:** Allowed; ignored during model inference
            - **Negative / NaN / Inf values:** Rejected with row-level error message
            """)

        sample = pd.DataFrame([
            {"Crop": "Rice",  "Crop_Year": 2018, "Season": "Kharif", "State": "Uttar Pradesh", "Area": 5000.0, "Annual_Rainfall": 1000.0, "Fertilizer": 500000.0, "Pesticide": 1000.0},
            {"Crop": "Wheat", "Crop_Year": 2019, "Season": "Rabi",   "State": "Punjab",         "Area": 8000.0, "Annual_Rainfall": 650.0,  "Fertilizer": 1200000.0,"Pesticide": 2500.0},
            {"Crop": "Maize", "Crop_Year": 2017, "Season": "Kharif", "State": "Karnataka",      "Area": 3500.0, "Annual_Rainfall": 850.0,  "Fertilizer": 300000.0, "Pesticide": 800.0},
        ])
        st.download_button(
            "📥 Download Sample CSV Template",
            data=sample.to_csv(index=False),
            file_name="cropzen_batch_sample.csv",
            mime="text/csv",
        )

    st.markdown("---")
    uploaded = st.file_uploader(
        T("batch_upload", lang),
        type=["csv"],
        help="Max file size: 5 MB. CSV format only. Up to 5,000 rows.",
    )

    if uploaded:
        file_size = len(uploaded.getvalue())
        batch_df, file_error = validate_batch_file(uploaded, uploaded.name, file_size)

        if file_error:
            st.error(f"❌ **{T('error', lang)}:** {file_error}")
            st.stop()

        present_cols, missing_cols, extra_cols = validate_batch_columns(batch_df)
        if missing_cols:
            st.error(f"❌ **{T('batch_error_cols', lang)}:** {', '.join(missing_cols)}")
            st.stop()

        if extra_cols:
            st.info(f"ℹ️ {len(extra_cols)} extra column(s) detected (`{', '.join(extra_cols)}`). Preserved but ignored during inference.")

        st.markdown(f'<div class="section-title">{T("batch_preview", lang)}</div>', unsafe_allow_html=True)
        st.dataframe(batch_df.head(5), use_container_width=True)

        col_l, col_r = st.columns(2)
        col_l.metric("Total Rows Uploaded", f"{len(batch_df):,}")
        col_r.metric("File Size", f"{file_size / 1024:.1f} KB")

        if st.button(f"🚀 {T('batch_run', lang)}", type="primary"):
            schema = load_schema_cached()
            if schema is None:
                st.error("Feature schema missing. Run train_model.py first.")
                st.stop()

            with st.spinner("Running predictions using active pipeline..."):
                result_df = predict_batch(batch_df, schema)

            ok_mask  = result_df["Prediction_Status"] == "OK"
            err_mask = result_df["Prediction_Status"] == "ERROR"

            c_v1, c_v2, c_v3 = st.columns(3)
            c_v1.metric(T("batch_valid_rows", lang),   f"{ok_mask.sum():,}")
            c_v2.metric(T("batch_invalid_rows", lang),  f"{err_mask.sum():,}")
            c_v3.metric("Validation Pass Rate", f"{(ok_mask.sum() / len(batch_df) * 100):.1f}%")

            if ok_mask.sum() > 0:
                st.markdown(f'<div class="section-title">{T("batch_results", lang)}</div>', unsafe_allow_html=True)
                st.caption("**Predicted Target:** `Yield` — historical estimate in **tonnes per hectare (t/ha)**.")
                display_cols = BATCH_REQUIRED_COLUMNS + ["Predicted_Yield"]
                st.dataframe(result_df.loc[ok_mask, display_cols], use_container_width=True)

                csv_out = result_df.loc[ok_mask, display_cols].to_csv(index=False)
                st.download_button(
                    f"📥 {T('batch_download', lang)} ({ok_mask.sum():,} records)",
                    data=csv_out,
                    file_name="cropzen_predictions.csv",
                    mime="text/csv",
                )

                st.markdown(f'<div class="section-title">{T("batch_summary", lang)}</div>', unsafe_allow_html=True)
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    fig = px.histogram(
                        result_df.loc[ok_mask], x="Predicted_Yield",
                        title="Distribution of Predicted Yields (t/ha)",
                        labels={"Predicted_Yield": "Predicted Yield (t/ha)"},
                        color_discrete_sequence=["#1F7A4D"],
                        template="plotly_white", nbins=30,
                    )
                    fig.update_layout(font=dict(color="#1F2937"), plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
                    st.plotly_chart(fig, use_container_width=True)

                with col_chart2:
                    if result_df.loc[ok_mask, "Crop"].nunique() > 1:
                        crop_avg = result_df.loc[ok_mask].groupby("Crop")["Predicted_Yield"].mean().reset_index()
                        fig_crop = px.bar(
                            crop_avg, x="Crop", y="Predicted_Yield",
                            title="Average Predicted Yield by Crop (t/ha)",
                            labels={"Predicted_Yield": "Avg Yield (t/ha)"},
                            color="Crop", template="plotly_white",
                        )
                        fig_crop.update_layout(font=dict(color="#1F2937"), plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
                        st.plotly_chart(fig_crop, use_container_width=True)
                    else:
                        pred_stats = result_df.loc[ok_mask, "Predicted_Yield"].describe().round(4)
                        st.dataframe(pred_stats.to_frame("Yield Summary (t/ha)"), use_container_width=True)

            if err_mask.sum() > 0:
                st.markdown(f'<div class="section-title">⚠️ Rejected Records ({err_mask.sum():,} rows)</div>',
                            unsafe_allow_html=True)
                st.markdown("The following records were rejected due to missing fields, unknown categories, or invalid values:")
                err_display_cols = ["Prediction_Error"] + [c for c in BATCH_REQUIRED_COLUMNS if c in result_df.columns]
                st.dataframe(result_df.loc[err_mask, err_display_cols], use_container_width=True)

                err_csv = result_df.loc[err_mask, err_display_cols].to_csv(index=False)
                st.download_button(
                    f"📥 Download Rejected Records ({err_mask.sum():,} rows)",
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
    render_page_header(
        "📊", T("explorer_title", lang),
        subtitle="Explore 19,689 historical Indian crop yield records with interactive filters and charts.",
        breadcrumb="CropZen › Dataset Explorer",
    )

    df = load_dataset()
    if df is None:
        st.error("Dataset not found at data/crop_yield.csv.")
        st.stop()

    # ── Filters (sidebar + reset button) ──────────────────────────────────────
    with st.sidebar:
        st.markdown('<hr style="border:none;border-top:1px solid #D1E7DA;margin:8px 0;" />', unsafe_allow_html=True)
        st.markdown("**🔍 Filters**")
        all_crops   = sorted(df["Crop"].unique().tolist())
        all_states  = sorted(df["State"].unique().tolist())
        all_seasons = sorted(df["Season"].unique().tolist())
        year_min, year_max = int(df["Crop_Year"].min()), int(df["Crop_Year"].max())

        sel_crops   = st.multiselect(T("explorer_filter_crop",   lang), all_crops,   default=[])
        sel_states  = st.multiselect(T("explorer_filter_state",  lang), all_states,  default=[])
        sel_seasons = st.multiselect(T("explorer_filter_season", lang), all_seasons, default=[])
        sel_years   = st.slider(T("explorer_filter_year", lang), year_min, year_max, (year_min, year_max))

        if st.button("↺ Reset All Filters", use_container_width=True):
            sel_crops   = []
            sel_states  = []
            sel_seasons = []
            sel_years   = (year_min, year_max)
            st.rerun()

    # Apply filters
    fdf = df.copy()
    if sel_crops:   fdf = fdf[fdf["Crop"].isin(sel_crops)]
    if sel_states:  fdf = fdf[fdf["State"].isin(sel_states)]
    if sel_seasons: fdf = fdf[fdf["Season"].isin(sel_seasons)]
    fdf = fdf[(fdf["Crop_Year"] >= sel_years[0]) & (fdf["Crop_Year"] <= sel_years[1])]

    # Filter counter
    filter_active = bool(sel_crops or sel_states or sel_seasons or sel_years != (year_min, year_max))
    st.markdown(
        f'<div class="info-notice">Showing <strong>{len(fdf):,}</strong> of <strong>{len(df):,}</strong> records'
        + (" (filters active)" if filter_active else " (all records)") + "</div>",
        unsafe_allow_html=True,
    )

    if fdf.empty:
        st.warning("⚠️ **No records match the selected filter combination.** Try widening your filters.")
        st.stop()

    # ── Overview metrics ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric(T("explorer_rows", lang), f"{len(fdf):,}")
    col2.metric(T("explorer_cols", lang), fdf.shape[1])
    col3.metric("Unique Crops in Selection", fdf["Crop"].nunique() if "Crop" in fdf.columns else "—")

    with st.expander("📋 Dataset Preview (first 20 rows)"):
        st.dataframe(fdf.head(20), use_container_width=True)

    # ── Missing value analysis ────────────────────────────────────────────────
    st.markdown(f'<div class="section-title">{T("explorer_missing", lang)}</div>', unsafe_allow_html=True)
    missing = fdf.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Count"]
    missing["Missing %"] = (missing["Missing Count"] / len(fdf) * 100).round(2)
    if missing["Missing Count"].sum() == 0:
        st.success("✅ No missing values in the filtered dataset.")
    else:
        st.dataframe(missing, use_container_width=True)

    # ── Descriptive stats ─────────────────────────────────────────────────────
    with st.expander(T("explorer_stats", lang)):
        num_cols = [c for c in ["Area", "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"] if c in fdf.columns]
        st.dataframe(fdf[num_cols].describe().round(4), use_container_width=True)

    # ── Yield distribution ────────────────────────────────────────────────────
    st.markdown(f'<div class="section-title">{T("explorer_distribution", lang)}</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    _chart_layout = dict(font=dict(color="#1F2937"), plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")

    with col_a:
        fig = px.histogram(
            fdf, x="Yield", nbins=50,
            title="Yield Distribution — Raw Values (t/ha)",
            labels={"Yield": "Yield (t/ha)"},
            color_discrete_sequence=["#1F7A4D"],
            template="plotly_white",
        )
        fig.update_layout(**_chart_layout)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fdf_pos = fdf[fdf["Yield"] > 0].copy()
        fdf_pos["log_yield"] = np.log1p(fdf_pos["Yield"])
        fig2 = px.histogram(
            fdf_pos, x="log_yield", nbins=50,
            title="Yield Distribution — Log Scale (excludes zero-yield records)",
            labels={"log_yield": "log(1 + Yield)"},
            color_discrete_sequence=["#2D9D63"],
            template="plotly_white",
        )
        fig2.update_layout(**_chart_layout)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Crop-wise comparison ──────────────────────────────────────────────────
    st.markdown(f'<div class="section-title">{T("explorer_crop_compare", lang)}</div>', unsafe_allow_html=True)
    top_n = min(20, fdf["Crop"].nunique())
    top_crops = fdf.groupby("Crop")["Yield"].median().nlargest(top_n).index.tolist()
    crop_df = fdf[fdf["Crop"].isin(top_crops)]
    fig3 = px.box(
        crop_df, x="Crop", y="Yield",
        title=f"Yield Distribution by Crop — Top {top_n} by Median Yield (t/ha)",
        labels={"Yield": "Yield (t/ha)"},
        color="Crop",
        color_discrete_sequence=px.colors.qualitative.Set3,
        template="plotly_white",
    )
    fig3.update_layout(showlegend=False, xaxis_tickangle=-45, **_chart_layout)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Season & Year ─────────────────────────────────────────────────────────
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f'<div class="section-title">{T("explorer_season_compare", lang)}</div>', unsafe_allow_html=True)
        fig4 = px.box(
            fdf, x="Season", y="Yield",
            title="Yield Distribution by Season (t/ha)",
            color="Season",
            color_discrete_sequence=["#1F7A4D", "#2D9D63", "#4DB87E", "#80CEA0", "#B3E2C7", "#E0F4EC"],
            template="plotly_white",
        )
        fig4.update_layout(showlegend=False, **_chart_layout)
        st.plotly_chart(fig4, use_container_width=True)

    with col_s2:
        st.markdown(f'<div class="section-title">{T("explorer_year_trend", lang)}</div>', unsafe_allow_html=True)
        yr_df = fdf.groupby("Crop_Year")["Yield"].median().reset_index()
        fig5 = px.line(
            yr_df, x="Crop_Year", y="Yield",
            title="Median Crop Yield per Year (t/ha)",
            markers=True,
            labels={"Yield": "Median Yield (t/ha)", "Crop_Year": "Year"},
            color_discrete_sequence=["#1F7A4D"],
            template="plotly_white",
        )
        fig5.update_layout(**_chart_layout)
        st.plotly_chart(fig5, use_container_width=True)

    # ── State-wise comparison ─────────────────────────────────────────────────
    st.markdown(f'<div class="section-title">{T("explorer_state_compare", lang)}</div>', unsafe_allow_html=True)
    state_df = fdf.groupby("State")["Yield"].median().reset_index().sort_values("Yield", ascending=True)
    fig6 = px.bar(
        state_df, x="Yield", y="State", orientation="h",
        title="Median Crop Yield by State (t/ha)",
        labels={"Yield": "Median Yield (t/ha)"},
        color="Yield", color_continuous_scale=["#C9E8D5", "#1F7A4D"],
        template="plotly_white",
    )
    fig6.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=40, b=0), **_chart_layout)
    st.plotly_chart(fig6, use_container_width=True)

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.markdown(f'<div class="section-title">{T("explorer_correlation", lang)}</div>', unsafe_allow_html=True)
    corr_cols = [c for c in ["Area", "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"] if c in fdf.columns]
    corr_matrix = fdf[corr_cols].corr()
    fig7 = px.imshow(
        corr_matrix,
        title="Pearson Correlation Matrix — Numerical Features",
        color_continuous_scale="RdYlGn",
        zmin=-1, zmax=1,
        aspect="auto",
        template="plotly_white",
        text_auto=".2f",
    )
    fig7.update_layout(**_chart_layout)
    st.plotly_chart(fig7, use_container_width=True)
    st.caption(T("explorer_note_corr", lang))

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        fig8 = px.scatter(
            fdf.sample(min(2000, len(fdf)), random_state=42),
            x="Annual_Rainfall", y="Yield",
            title="Annual Rainfall vs Yield — Sample of 2,000 Records",
            labels={"Annual_Rainfall": "Annual Rainfall (mm)", "Yield": "Yield (t/ha)"},
            color="Season", opacity=0.6,
            template="plotly_white",
        )
        fig8.update_layout(**_chart_layout)
        st.plotly_chart(fig8, use_container_width=True)

    with col_r2:
        fig9 = px.scatter(
            fdf.sample(min(2000, len(fdf)), random_state=42),
            x="Area", y="Yield",
            title="Cultivated Area vs Yield — Sample of 2,000 Records (log-X)",
            labels={"Area": "Area (ha)", "Yield": "Yield (t/ha)"},
            color="Season", opacity=0.6,
            log_x=True,
            template="plotly_white",
        )
        fig9.update_layout(**_chart_layout)
        st.plotly_chart(fig9, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# E. MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "performance":
    render_page_header(
        "📈", T("perf_title", lang),
        subtitle=T("perf_subtitle", lang),
        breadcrumb="CropZen › Model Performance",
    )

    eval_meta = load_eval_cached()
    if eval_meta is None:
        st.error(T("home_model_missing", lang))
        st.code(".venv\\Scripts\\python.exe train_model.py", language="bash")
        st.stop()

    metrics_all = eval_meta.get("metrics", {})
    best_model  = eval_meta.get("best_model", "")
    _chart_layout = dict(font=dict(color="#1F2937"), plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")

    # ── Best model summary ────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="card card-accent">
      <div style="font-size:0.8rem;color:#6B7280;margin-bottom:4px;">Best Performing Model (lowest MAE on held-out test set)</div>
      <div style="font-size:1.4rem;font-weight:700;color:#1F7A4D;">⭐ {best_model}</div>
      <div style="font-size:0.82rem;color:#6B7280;margin-top:4px;">
        Train: {eval_meta.get('train_size', 0):,} rows &nbsp;|&nbsp;
        Test: {eval_meta.get('test_size', 0):,} rows &nbsp;|&nbsp;
        Split: {eval_meta.get('split_method', '80/20 random')}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics table ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">All Models — Evaluation Metrics (Held-Out Test Set)</div>',
                unsafe_allow_html=True)
    rows = []
    for name, m in metrics_all.items():
        rows.append({
            "Model": name,
            "MAE (t/ha)":  round(m.get("MAE",  0), 4) if m.get("MAE")  is not None else None,
            "RMSE (t/ha)": round(m.get("RMSE", 0), 4) if m.get("RMSE") is not None else None,
            "R²":          round(m.get("R2",   0), 4) if m.get("R2")   is not None else None,
            "Selected":    "⭐ Best" if name == best_model else "",
        })
    metrics_df = pd.DataFrame(rows)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # ── Bar charts ────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    _bar_colors = px.colors.qualitative.Set2

    with col1:
        fig = px.bar(
            metrics_df, x="Model", y="MAE (t/ha)",
            title=T("perf_mae", lang),
            color="Model", color_discrete_sequence=_bar_colors,
            template="plotly_white",
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-20, **_chart_layout)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(T("perf_mae_explain", lang))

    with col2:
        fig2 = px.bar(
            metrics_df, x="Model", y="RMSE (t/ha)",
            title=T("perf_rmse", lang),
            color="Model", color_discrete_sequence=_bar_colors,
            template="plotly_white",
        )
        fig2.update_layout(showlegend=False, xaxis_tickangle=-20, **_chart_layout)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(T("perf_rmse_explain", lang))

    with col3:
        fig3 = px.bar(
            metrics_df, x="Model", y="R²",
            title=T("perf_r2", lang),
            color="Model", color_discrete_sequence=_bar_colors,
            template="plotly_white",
        )
        fig3.update_layout(showlegend=False, xaxis_tickangle=-20, **_chart_layout)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(T("perf_r2_explain", lang))

    # ── Actual vs Predicted ───────────────────────────────────────────────────
    st.markdown(f'<div class="section-title">{T("perf_actual_vs_pred", lang)}</div>', unsafe_allow_html=True)
    y_true = eval_meta.get("y_true_test", [])
    y_pred = eval_meta.get("y_pred_test", [])

    if y_true and y_pred:
        scatter_df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
        if len(scatter_df) > 2000:
            scatter_df = scatter_df.sample(2000, random_state=42)

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            max_val = max(scatter_df["Actual"].max(), scatter_df["Predicted"].max())
            fig4 = px.scatter(
                scatter_df, x="Actual", y="Predicted",
                title=f"Actual vs Predicted — {best_model} (sample of 2,000)",
                labels={"Actual": "Actual Yield (t/ha)", "Predicted": "Predicted Yield (t/ha)"},
                opacity=0.4,
                color_discrete_sequence=["#1F7A4D"],
                template="plotly_white",
            )
            # Perfect-prediction diagonal
            fig4.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines", name="Perfect Prediction (y = x)",
                line=dict(color="#DC2626", dash="dash", width=1.5),
            ))
            fig4.update_layout(**_chart_layout)
            st.plotly_chart(fig4, use_container_width=True)
            st.caption("Points close to the red diagonal line indicate accurate predictions. Clusters far from the line indicate systematic errors.")

        with col_p2:
            scatter_df["Residual"] = scatter_df["Predicted"] - scatter_df["Actual"]
            fig5 = px.scatter(
                scatter_df, x="Predicted", y="Residual",
                title=f"Residuals Plot — {best_model}",
                labels={"Predicted": "Predicted Yield (t/ha)", "Residual": "Residual (Predicted − Actual)"},
                opacity=0.4,
                color_discrete_sequence=["#3B82F6"],
                template="plotly_white",
            )
            fig5.add_hline(y=0, line_dash="dash", line_color="#DC2626", line_width=1.5)
            fig5.update_layout(**_chart_layout)
            st.plotly_chart(fig5, use_container_width=True)
            st.caption("Residuals should be randomly scattered around zero. Patterns suggest model bias. The large tail above the line reflects extreme-yield crops (e.g. Coconut).")

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown(f'<div class="section-title">{T("perf_feat_importance", lang)}</div>', unsafe_allow_html=True)
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
                color="Importance",
                color_continuous_scale=["#C9E8D5", "#1F7A4D"],
                template="plotly_white",
            )
            fig6.update_layout(coloraxis_showscale=False, **_chart_layout)
            st.plotly_chart(fig6, use_container_width=True)
            st.caption(T("perf_feat_note", lang))
        else:
            st.info("Feature importance is not available for Linear Regression (no native importances).")

    # ── Statistical notes ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🔍 Evaluation Methodology & Metric Interpretation</div>',
                unsafe_allow_html=True)
    col_note1, col_note2 = st.columns(2)
    with col_note1:
        st.markdown("""
        <div class="card">
          <strong style="color:#1F7A4D;">Understanding MAE vs RMSE Disparity</strong>
          <ul style="margin-top:8px;font-size:0.85rem;color:#374151;">
            <li><strong>MAE ≈ 15.47 t/ha:</strong> Average absolute deviation. Typical crop errors are relatively compact.</li>
            <li><strong>RMSE ≈ 297 t/ha:</strong> RMSE squares errors before averaging — it is heavily weighted by extreme-yield crops like <em>Coconut</em> (raw count units up to 21,105). This inflates RMSE without representing typical error.</li>
            <li><strong>R² = 0.8898:</strong> The Decision Tree explains ~89% of variance in held-out records. <em>R² is not classification accuracy.</em></li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_note2:
        st.markdown(f"""
        <div class="card">
          <strong style="color:#1F7A4D;">Train / Test Split Design</strong>
          <ul style="margin-top:8px;font-size:0.85rem;color:#374151;">
            <li><strong>Strategy:</strong> {eval_meta.get('split_method', 'Random 80/20, seed=42')}</li>
            <li><strong>Test set:</strong> {eval_meta.get('test_size', 0):,} records ({eval_meta.get('test_size_fraction', 0.2)*100:.0f}%)</li>
            <li><strong>Limitation:</strong> {eval_meta.get('split_limitation', 'Random split does not guarantee chronological generalization.')}</li>
            <li><strong>Trained at:</strong> {eval_meta.get('trained_at', 'N/A')}</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    st.caption("All metrics above are loaded directly from verified evaluation artifacts produced by train_model.py.")

    # ── Model descriptions ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Model Descriptions</div>', unsafe_allow_html=True)
    descriptions = eval_meta.get("model_descriptions", {})
    for mname, mdesc in descriptions.items():
        badge = "⭐ " if mname == best_model else ""
        st.markdown(f"**{badge}{mname}**: {mdesc}")


# ══════════════════════════════════════════════════════════════════════════════
# F. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "training":
    render_page_header(
        "⚙️", T("train_title", lang),
        subtitle=T("train_subtitle", lang),
        breadcrumb="CropZen › Model Training",
    )

    eval_meta = load_eval_cached()

    # ── Current model status ───────────────────────────────────────────────────
    st.markdown(f'<div class="section-title">{T("train_status", lang)}</div>', unsafe_allow_html=True)
    if eval_meta:
        best = eval_meta.get("best_model", "")
        trained_at = eval_meta.get("trained_at", "Unknown")
        m = eval_meta.get("metrics", {}).get(best, {})
        st.markdown(f"""
        <div class="card card-accent">
          <div style="font-size:0.8rem;color:#6B7280;margin-bottom:6px;">{T('train_last_run', lang)}: {trained_at}</div>
          <div style="font-size:1.2rem;font-weight:700;color:#1F7A4D;">✅ {T('train_success', lang)}</div>
          <div style="font-size:1rem;color:#374151;margin-top:4px;">Best Model: <strong>{best}</strong></div>
        </div>
        """, unsafe_allow_html=True)

        if m:
            c1, c2, c3 = st.columns(3)
            c1.metric("R²",   f"{m.get('R2',   0):.4f}")
            c2.metric("MAE",  f"{m.get('MAE',  0):.4f}")
            c3.metric("RMSE", f"{m.get('RMSE', 0):.4f}")
    else:
        st.warning(T("home_model_missing", lang))

    # ── Retraining disabled (security) ────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">Local Retraining Guide</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="security-note">
      <strong>🔒 Public Retraining Disabled</strong><br>
      The in-app retraining button is intentionally disabled on the deployed application because a public
      Streamlit app cannot distinguish the project owner from ordinary visitors. Running arbitrary
      retraining jobs server-side would pose a security and resource risk.<br><br>
      <strong>To retrain locally:</strong> follow the steps below.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Step-by-step local retraining:**

    **1.** Activate the project virtual environment:
    ```powershell
    .venv\\Scripts\\Activate.ps1
    ```

    **2.** Run the training script:
    ```powershell
    .venv\\Scripts\\python.exe train_model.py
    ```

    **3.** Verify the output artifacts exist:
    ```powershell
    ls models/best_model.pkl
    ls outputs/evaluation.json
    ls outputs/feature_schema.json
    ```

    **4.** Restart the Streamlit app to load the new model:
    ```powershell
    .venv\\Scripts\\python.exe -m streamlit run app.py
    ```

    The last known-good model remains active if a new training run fails.
    """)

    # Show current model file info
    model_path = ROOT / "models" / "best_model.pkl"
    if model_path.exists():
        size_kb = model_path.stat().st_size / 1024
        st.info(f"📦 Model file: `models/best_model.pkl` &nbsp;({size_kb:.1f} KB)")

    schema_path = ROOT / "outputs" / "feature_schema.json"
    if schema_path.exists():
        st.info("📋 Feature schema: `outputs/feature_schema.json`")


# ══════════════════════════════════════════════════════════════════════════════
# G. PREDICTION HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "history":
    render_page_header(
        "🕐", T("history_title", lang),
        subtitle=T("history_subtitle", lang),
        breadcrumb="CropZen › Prediction History",
    )

    st.markdown("""
    <div class="info-notice">
      ℹ️ History is stored only for this browser session and is cleared when you close or refresh the tab.
      Download a CSV to preserve your results permanently.
    </div>
    """, unsafe_allow_html=True)

    history = st.session_state.prediction_history

    if not history:
        st.info(T("history_empty", lang))
    else:
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

        if len(history) > 1:
            fig = px.bar(
                hist_df, x="timestamp", y="predicted_yield",
                title="Predicted Yields Over This Session",
                color="crop",
                labels={"predicted_yield": "Yield (t/ha)", "timestamp": "Time"},
                template="plotly_white",
            )
            fig.update_layout(xaxis_tickangle=-30,
                              font=dict(color="#1F2937"),
                              plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
            st.plotly_chart(fig, use_container_width=True)

        col_dl, col_clear = st.columns([3, 1])
        with col_dl:
            st.download_button(
                "📥 Download History CSV",
                data=hist_df.to_csv(index=False),
                file_name="prediction_history.csv",
                mime="text/csv",
            )
        with col_clear:
            if st.button("🗑️ " + T("history_clear", lang), type="secondary"):
                st.session_state["_confirm_clear"] = True

        if st.session_state.get("_confirm_clear", False):
            st.warning("Are you sure you want to clear all prediction history? This cannot be undone.")
            c_yes, c_no = st.columns(2)
            with c_yes:
                if st.button("✅ Yes, clear history", type="primary"):
                    st.session_state.prediction_history = []
                    st.session_state["_confirm_clear"] = False
                    st.success(T("history_cleared", lang))
                    st.rerun()
            with c_no:
                if st.button("❌ Cancel"):
                    st.session_state["_confirm_clear"] = False
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# H. ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "about":
    render_page_header(
        "ℹ️", T("about_title", lang),
        subtitle="5th Semester CSE Design Project — Academic Use",
        breadcrumb="CropZen › About",
    )

    # Prominent limitations card at top
    st.markdown("""
    <div class="limitation-alert">
      <strong>⚠️ Important Limitations — Please Read Before Use</strong><br>
      This application is an <strong>academic research prototype</strong> built on historical Indian agricultural data (1997–2020).
      Predictions are empirical estimates from trained ML models and do <strong>NOT</strong> constitute certified agricultural advisory,
      guaranteed yields, or vetted farming recommendations. Do not use these predictions for actual crop planning, financial decisions,
      or government reporting without independent expert verification.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Overview", "📂 Dataset & Provenance", "🤖 Algorithms", "📏 Metrics", "⚖️ Limitations & Ethics"
    ])

    with tab1:
        st.markdown(f"### {T('about_problem', lang)}")
        st.markdown("""
        Crop yield prediction is challenging because agricultural production varies with environmental conditions,
        crop characteristics, and regional differences. The reliability of machine-learning predictions depends
        on data quality, feature selection, and model performance.

        This project develops a web-based system that preprocesses historical agricultural data, trains and compares
        multiple regression models, and predicts crop yield using the features supported by the dataset. The system
        evaluates predictions using MAE, RMSE and R² and communicates the limitations of the resulting estimates.
        """)

        st.markdown(f"### {T('about_objectives', lang)}")
        st.markdown("""
        1. Perform exploratory data analysis on Indian crop yield data (1997–2020).
        2. Train and compare Linear Regression, Decision Tree and Random Forest models.
        3. Evaluate models objectively on a held-out test set (80/20, seed=42).
        4. Build an interactive prediction interface with English and Hindi support.
        5. Provide voice input and spoken output using browser Web Speech APIs.
        6. Enable batch predictions via CSV upload (max 5 MB / 5,000 rows).
        7. Explain model behaviour using feature importance.
        """)

        st.markdown(f"### {T('about_scope', lang)}")
        col_impl, col_future = st.columns(2)
        with col_impl:
            st.markdown("""
            **Implemented:**
            - ✅ Data audit and EDA
            - ✅ ML training pipeline (3 models)
            - ✅ Single-record prediction
            - ✅ Batch CSV prediction
            - ✅ Dataset explorer with filters
            - ✅ Model performance dashboard
            - ✅ Feature importance
            - ✅ English / Hindi interface
            - ✅ Voice input and spoken output
            - ✅ Session prediction history
            - ✅ Responsive design
            """)
        with col_future:
            st.markdown(f"""
            **{T('about_future', lang)}:**
            - ❌ Satellite or remote sensing data
            - ❌ Live weather API integration
            - ❌ IoT sensor input
            - ❌ Soil composition features
            - ❌ Irrigation / fertilizer recommendations
            - ❌ Persistent user database
            - ❌ Multi-year forecasting
            - ❌ Group-based train/test split
            - ❌ SHAP-based model explanations
            """)

    with tab2:
        st.markdown(f"### {T('about_dataset', lang)}")
        st.markdown("""
        **Source:** Historical Indian agricultural crop production dataset (1997–2020), derived from
        Ministry of Agriculture and Farmers Welfare public records / Kaggle open repositories.

        **License:** Open educational dataset for non-commercial academic research.
        """)

        st.markdown("""
        | Property | Value |
        |----------|-------|
        | Records | 19,689 rows |
        | Columns | 10 |
        | Unique Crops | 55 |
        | States Covered | 30 |
        | Agricultural Seasons | 6 (Kharif, Rabi, Whole Year, Summer, Autumn, Winter) |
        | Historical Years | 1997–2020 |
        """)

        st.markdown("""
        **Validated Feature Schema:**
        - **Input Features:** `Crop`, `Season`, `State`, `Crop_Year`, `Area` (ha), `Annual_Rainfall` (mm), `Fertilizer` (kg), `Pesticide` (kg)
        - **Target:** `Yield` (log1p-transformed during training; expm1 inverse applied to predictions; metric tonnes/ha)

        **Target Leakage Exclusion:** `Production` is strictly excluded because it represents post-harvest output
        and in many records correlates directly with the target computation (`Yield = Production / Area`),
        causing artificial target leakage.

        **Empirical Evaluation Design:**
        - Fixed 80/20 train/test split (seed=42) on 19,689 records (15,751 train / 3,938 test)
        - 112 zero-yield records preserved as genuine historical crop failure events
        - Tail crops (e.g., Coconut counted in units) inflate raw RMSE significantly
        """)

    with tab3:
        st.markdown(f"### {T('about_algorithms', lang)}")

        col_lr, col_dt, col_rf = st.columns(3)
        with col_lr:
            st.markdown("""
            <div class="card">
              <strong style="color:#1F7A4D;">Linear Regression</strong>
              <p style="font-size:0.85rem;color:#374151;margin-top:8px;">
                Fits a linear relationship between input features and yield.
                Simple, fast, and interpretable. Struggles with non-linear patterns.
                Acts as a baseline for comparison.
              </p>
            </div>
            """, unsafe_allow_html=True)
        with col_dt:
            st.markdown("""
            <div class="card">
              <strong style="color:#1F7A4D;">Decision Tree Regressor</strong>
              <p style="font-size:0.85rem;color:#374151;margin-top:8px;">
                Learns recursive splits on feature values. Captures non-linear patterns.
                Prone to overfitting without depth limits.
                Provides native feature importance scores.
              </p>
            </div>
            """, unsafe_allow_html=True)
        with col_rf:
            st.markdown("""
            <div class="card">
              <strong style="color:#1F7A4D;">Random Forest Regressor</strong>
              <p style="font-size:0.85rem;color:#374151;margin-top:8px;">
                Ensemble of 100 decision trees. More robust to overfitting.
                Provides averaged feature importance. Slower to train and predict.
              </p>
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.markdown(f"### {T('about_metrics', lang)}")
        st.markdown("""
        | Metric | Full Name | Meaning | Better When |
        |--------|-----------|---------|-------------|
        | MAE | Mean Absolute Error | Average absolute difference between predicted and actual yield | Lower ↓ |
        | RMSE | Root Mean Squared Error | Like MAE but penalises large errors heavily | Lower ↓ |
        | R² | Coefficient of Determination | Proportion of variance explained by the model (max 1.0) | Higher ↑ |

        > **R² is NOT classification accuracy.** A value of 0.89 means the model explains 89% of the
        variance in yield across the test set — it does not mean 89% of predictions are "correct".
        """)

    with tab5:
        st.markdown("### ⚖️ Scope, Limitations & Ethics")
        st.markdown("""
        <div class="limitation-alert" style="margin-top:12px;">
          This system is a <strong>research and education prototype</strong>. The following limitations apply:
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        **Data Limitations:**
        - Dataset covers Indian states up to 2020. Post-2020 conditions (climate change, new varieties, policy changes) are not represented.
        - 112 zero-yield records may reflect data quality issues or genuine crop failures — the model does not distinguish.
        - Tail-distribution crops (e.g., Coconut, Sugarcane) have extreme raw yield values that inflate RMSE metrics.

        **Model Limitations:**
        - Random 80/20 split does not guarantee generalization to future years or unseen state-crop combinations.
        - The model has no awareness of weather forecasts, soil quality, irrigation, or market conditions.
        - Feature encoding encodes the mean response for unseen categories — predictions for new crops or states will be biased.

        **Ethical Considerations:**
        - Do not use predictions for actual crop planning, financial decisions, insurance claims, or government reporting.
        - The model reflects historical patterns which may embed regional or crop-type biases.
        - Predictions in underrepresented crop-state-season combinations will have higher uncertainty.
        - This system does not provide agronomic advice and should not replace expert agricultural consultation.
        """)
