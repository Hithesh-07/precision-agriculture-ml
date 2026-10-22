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
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from src.data_processing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    DATA_PATH,
    DATA_PATH_MODEL_READY,
    DATA_PATH_DISTRICT,
    DATA_PATH_OFFICIAL_2021_25,
    DATA_PATH_COMBINED,
    DATASET_REGISTRY,
    NUMERIC_FEATURES,
    TARGET,
    check_out_of_training_range,
    load_raw_data,
    load_model_ready_data,
    load_district_data,
    load_official_2021_25_data,
    load_combined_data,
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
from src.multilingual_voice import (
    build_multilingual_result_card,
    build_multilingual_spoken_answer,
    detect_language,
    parse_multilingual_voice_text,
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

# ── session state initialisation ──────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""
if "user_spoken_lang" not in st.session_state:
    st.session_state.user_spoken_lang = "en"

lang = st.session_state.lang
theme = st.session_state.theme
is_dark = (theme == "dark")

# Palette & CSS Tokens for Light vs Dark mode (World-Class AgriTech Design System)
if is_dark:
    bg_app = "#0B1120"           # Deep space navy canvas
    bg_sidebar = "#0D1527"       # Structured sidebar background
    bg_card = "#111C32"          # Elevated surface card
    bg_card_hover = "#162542"    # Micro-interaction card hover
    bg_meta_item = "rgba(255, 255, 255, 0.04)"
    border_color = "rgba(255, 255, 255, 0.08)"
    border_hover = "rgba(34, 197, 94, 0.4)"
    text_main = "#F8FAFC"        # High-contrast primary text
    text_muted = "#94A3B8"       # Secondary text
    text_subdued = "#64748B"     # Tertiary caption text
    accent_green = "#10B981"     # Vibrant emerald
    accent_green_hover = "#059669"
    accent_glow = "rgba(16, 185, 129, 0.25)"
    topbar_bg = "#0D1527"
    banner_grad = "radial-gradient(100% 120% at 20% 0%, #064E3B 0%, #0B1120 75%), linear-gradient(135deg, #064E3B 0%, #0F172A 100%)"
    banner_border = "rgba(16, 185, 129, 0.3)"
    banner_shadow = "0 12px 32px -4px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1)"
    radio_hover = "rgba(255, 255, 255, 0.05)"
    radio_active = "#162542"
    badge_ok_bg = "rgba(16, 185, 129, 0.15)"
    badge_ok_text = "#34D399"
    badge_warn_bg = "rgba(245, 158, 11, 0.15)"
    badge_warn_text = "#FBBF24"
    badge_info_bg = "rgba(56, 189, 248, 0.15)"
    badge_info_text = "#38BDF8"
    badge_err_bg = "rgba(239, 68, 68, 0.15)"
    badge_err_text = "#F87171"
    disclaimer_bg = "rgba(245, 158, 11, 0.08)"
    disclaimer_border = "#F59E0B"
    disclaimer_text = "#FDE68A"
    card_shadow = "0 4px 20px -2px rgba(0, 0, 0, 0.35), 0 2px 6px -1px rgba(0, 0, 0, 0.2)"
    plotly_template = "plotly_dark"
    plotly_bg = "#111C32"
else:
    bg_app = "#F8FAFC"           # Crisp modern light canvas
    bg_sidebar = "#F1F5F9"       # Refined slate-50 sidebar
    bg_card = "#FFFFFF"          # Pure white surface card
    bg_card_hover = "#F8FAFC"    # Subtle hover tint
    bg_meta_item = "rgba(0, 0, 0, 0.03)"
    border_color = "rgba(0, 0, 0, 0.08)"
    border_hover = "#10B981"
    text_main = "#0F172A"        # Deep slate text
    text_muted = "#475569"       # Muted slate
    text_subdued = "#94A3B8"     # Subdued caption
    accent_green = "#059669"     # Refined emerald-600
    accent_green_hover = "#047857"
    accent_glow = "rgba(5, 150, 105, 0.15)"
    topbar_bg = "#FFFFFF"
    banner_grad = "linear-gradient(135deg, #059669 0%, #047857 50%, #064E3B 100%)"
    banner_border = "#059669"
    banner_shadow = "0 10px 25px -4px rgba(4, 120, 87, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.2)"
    radio_hover = "rgba(0, 0, 0, 0.04)"
    radio_active = "#E2E8F0"
    badge_ok_bg = "#DCFCE7"
    badge_ok_text = "#166534"
    badge_warn_bg = "#FEF9C3"
    badge_warn_text = "#854D0E"
    badge_info_bg = "#E0F2FE"
    badge_info_text = "#075985"
    badge_err_bg = "#FEE2E2"
    badge_err_text = "#991B1B"
    disclaimer_bg = "#FFFBEB"
    disclaimer_border = "#F59E0B"
    disclaimer_text = "#92400E"
    card_shadow = "0 1px 3px 0 rgba(0, 0, 0, 0.06), 0 1px 2px -1px rgba(0, 0, 0, 0.06)"
    plotly_template = "plotly_white"
    plotly_bg = "#FFFFFF"

# ── Design System CSS ─────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Typography & Foundation ───────────────────────────────────────────────── */
html, body, [class*="css"] {{
  font-family: -apple-system, BlinkMacSystemFont, "Inter", "SF Pro Display", "Segoe UI", Roboto, sans-serif;
  color: {text_main};
  letter-spacing: -0.011em;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}}

/* ── Streamlit native top header & toolbar (Flawless Dark & Light) ──────────── */
header,
header[data-testid="stHeader"],
[data-testid="stHeader"],
.stApp > header,
.stAppHeader,
div[data-testid="stHeader"],
div[data-testid="stToolbar"] {{
  background-color: {bg_app} !important;
  background: {bg_app} !important;
  color: {text_main} !important;
  border-bottom: 1px solid {border_color} !important;
}}
header::before,
header::after,
[data-testid="stHeader"]::before,
[data-testid="stHeader"]::after {{
  background: {bg_app} !important;
}}
header *,
[data-testid="stHeader"] *,
.stAppHeader *,
[data-testid="stToolbar"] * {{
  color: {text_main} !important;
  fill: {text_main} !important;
}}
[data-testid="stToolbar"] {{
  background-color: transparent !important;
  background: transparent !important;
  color: {text_main} !important;
}}
[data-testid="stDecoration"] {{
  display: none !important;
  visibility: hidden !important;
  height: 0px !important;
}}

/* ── Page Canvas Background ───────────────────────────────────────────────── */
.stApp {{
  background-color: {bg_app} !important;
  background: {bg_app} !important;
  color: {text_main} !important;
}}

/* ── Block Container (Fluid Spacing & Max Width) ───────────────────────────── */
.block-container {{
  padding-top: 1.5rem !important;
  padding-bottom: 3.5rem !important;
  max-width: 1200px;
}}

/* ── Sidebar (Kowalski / SaaS Sidebar Architecture) ────────────────────────── */
[data-testid="stSidebar"] {{
  background-color: {bg_sidebar} !important;
  border-right: 1px solid {border_color} !important;
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown {{
  color: {text_main} !important;
}}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label {{
  color: {text_main} !important;
  font-weight: 600;
  font-size: 0.82rem;
  letter-spacing: 0.01em;
}}
[data-testid="stSidebar"] [data-baseweb="select"] * {{
  color: {text_main} !important;
}}
/* Navigation radio — Modern tactile pills with subtle active indicator */
[data-testid="stSidebar"] [data-testid="stRadio"] label {{
  padding: 9px 14px !important;
  border-radius: 9px !important;
  margin: 3px 0 !important;
  display: flex !important;
  align-items: center !important;
  transition: all 0.15s cubic-bezier(0.16, 1, 0.3, 1);
  cursor: pointer;
  font-size: 0.88rem;
  border: 1px solid transparent !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
  background: {radio_hover} !important;
  border-color: {border_color} !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] + div label,
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
  background: {radio_active} !important;
  border: 1px solid {border_color} !important;
  border-left: 3px solid {accent_green} !important;
  font-weight: 600 !important;
  color: {accent_green} !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label span {{
  color: {text_main} !important;
}}

/* ── Page Header Block ─────────────────────────────────────────────────────── */
.page-title {{
  font-size: 1.75rem;
  font-weight: 700;
  letter-spacing: -0.025em;
  color: {text_main};
  margin-bottom: 0.35rem;
  line-height: 1.25;
}}
.page-subtitle {{
  font-size: 0.95rem;
  color: {text_muted};
  margin-bottom: 1.5rem;
  line-height: 1.5;
}}
.page-breadcrumb {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 0.76rem;
  font-weight: 500;
  color: {text_muted};
  padding: 3px 10px;
  border-radius: 9999px;
  background: {bg_meta_item};
  border: 1px solid {border_color};
  margin-bottom: 0.75rem;
  letter-spacing: 0.02em;
}}

/* ── Section Dividers ──────────────────────────────────────────────────────── */
.section-title {{
  font-size: 1.15rem;
  font-weight: 700;
  letter-spacing: -0.015em;
  color: {text_main};
  margin-top: 1.8rem;
  margin-bottom: 0.85rem;
  display: flex;
  align-items: center;
  gap: 8px;
}}
.section-title::after {{
  content: "";
  flex: 1;
  height: 1px;
  background: {border_color};
  margin-left: 8px;
}}

/* ── Surface Cards (Glassmorphism & Depth) ─────────────────────────────────── */
.card {{
  background: {bg_card};
  border: 1px solid {border_color};
  border-radius: 14px;
  padding: 22px 24px;
  margin-bottom: 18px;
  box-shadow: {card_shadow};
  backdrop-filter: blur(12px);
  color: {text_main};
  transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.15s ease;
}}
.card:hover {{
  border-color: {border_hover};
}}
.card-accent {{
  border-left: 4px solid {accent_green} !important;
}}

/* ── Hero Banner (Awwwards / High-Impact Value Proposition) ────────────────── */
.hero-banner {{
  background: {banner_grad};
  border-radius: 18px;
  padding: 38px 42px;
  margin-bottom: 26px;
  color: #FFFFFF;
  border: 1px solid {banner_border};
  box-shadow: {banner_shadow};
  position: relative;
  overflow: hidden;
}}
.hero-banner h1 {{
  font-size: 2.2rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  color: #FFFFFF !important;
  line-height: 1.2;
  margin-bottom: 12px;
}}
.hero-banner p {{
  font-size: 1.02rem;
  line-height: 1.6;
  color: rgba(255, 255, 255, 0.92) !important;
  margin-bottom: 0;
  max-width: 840px;
}}
.hero-badge-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 20px;
}}
.hero-badge {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border-radius: 9999px;
  background: rgba(255, 255, 255, 0.12);
  border: 1px solid rgba(255, 255, 255, 0.2);
  backdrop-filter: blur(8px);
  font-size: 0.78rem;
  font-weight: 600;
  color: #FFFFFF;
  letter-spacing: 0.02em;
}}

/* ── Action Cards (Home Quick Actions) ─────────────────────────────────────── */
.action-card {{
  background: {bg_card};
  border: 1px solid {border_color};
  border-radius: 14px;
  padding: 24px 20px;
  text-align: center;
  box-shadow: {card_shadow};
  transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
  cursor: pointer;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}}
.action-card:hover {{
  box-shadow: 0 12px 28px -4px rgba(16, 185, 129, 0.18);
  border-color: {accent_green};
  transform: translateY(-3px);
}}
.action-card .action-icon {{
  font-size: 2.4rem;
  margin-bottom: 12px;
  display: inline-block;
  filter: drop-shadow(0 2px 8px rgba(0,0,0,0.15));
}}
.action-card .action-title {{
  font-size: 1.05rem;
  font-weight: 700;
  letter-spacing: -0.015em;
  color: {text_main};
  margin-bottom: 8px;
}}
.action-card .action-desc {{
  font-size: 0.85rem;
  line-height: 1.5;
  color: {text_muted};
  flex-grow: 1;
}}

/* ── Stat Cards (Metrics Grid) ─────────────────────────────────────────────── */
.stat-card {{
  background: {bg_card};
  border: 1px solid {border_color};
  border-radius: 14px;
  padding: 18px 20px;
  text-align: center;
  box-shadow: {card_shadow};
  transition: transform 0.15s ease, border-color 0.15s ease;
}}
.stat-card:hover {{
  transform: translateY(-2px);
  border-color: {border_hover};
}}
.stat-card .stat-value {{
  font-size: 2.1rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  color: {accent_green};
  line-height: 1.15;
}}
.stat-card .stat-label {{
  font-size: 0.74rem;
  font-weight: 600;
  color: {text_muted};
  margin-top: 6px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}}

/* ── Step Guide & Workflows ────────────────────────────────────────────────── */
.steps-guide {{
  background: {bg_card};
  border: 1px solid {border_color};
  border-left: 4px solid {accent_green};
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 20px;
  font-size: 0.9rem;
  line-height: 1.6;
  color: {text_main};
  box-shadow: {card_shadow};
}}
.steps-guide strong {{
  color: {accent_green};
}}

/* ── Interactive 4-Step Pipeline Flow ──────────────────────────────────────── */
.pipeline-grid {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin: 18px 0;
}}
@media (max-width: 900px) {{
  .pipeline-grid {{
    grid-template-columns: 1fr;
  }}
}}
.pipeline-step {{
  background: {bg_card};
  border: 1px solid {border_color};
  border-radius: 12px;
  padding: 20px 18px;
  box-shadow: {card_shadow};
  display: flex;
  flex-direction: column;
}}
.pipeline-step-header {{
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}}
.pipeline-step-num {{
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: {accent_green};
  color: #FFFFFF;
  font-weight: 700;
  font-size: 0.85rem;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
  flex-shrink: 0;
}}
.pipeline-step-title {{
  font-weight: 700;
  font-size: 0.92rem;
  color: {text_main};
  letter-spacing: -0.01em;
}}
.pipeline-step-desc {{
  font-size: 0.82rem;
  line-height: 1.5;
  color: {text_muted};
}}

/* ── High-Tier Diagnostic Result Card (Meng To / Kowalski) ─────────────────── */
.result-card {{
  background: {bg_card};
  border: 2px solid {accent_green};
  border-radius: 16px;
  padding: 32px 36px;
  margin: 22px 0;
  text-align: center;
  box-shadow: 0 10px 32px -4px {accent_glow};
  position: relative;
  overflow: hidden;
}}
.result-card .yield-number {{
  font-size: 3.8rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  color: {accent_green};
  line-height: 1;
  margin: 8px 0;
  text-shadow: 0 0 24px {accent_glow};
}}
.result-card .yield-unit {{
  font-size: 1rem;
  font-weight: 600;
  color: {text_main};
  margin-top: 6px;
}}
.result-card .result-context {{
  font-size: 0.88rem;
  color: {text_muted};
  margin-top: 8px;
}}
.result-card .result-meta {{
  margin-top: 22px;
  padding-top: 18px;
  border-top: 1px dashed {border_color};
  display: flex;
  justify-content: center;
  gap: 16px;
  flex-wrap: wrap;
}}
.result-card .result-meta-item {{
  background: {bg_meta_item};
  border: 1px solid {border_color};
  border-radius: 8px;
  padding: 6px 14px;
  font-size: 0.82rem;
  color: {text_muted};
}}
.result-card .result-meta-item strong {{
  color: {text_main};
}}

/* ── Badges (Apple-Style Capsule Pills) ────────────────────────────────────── */
.badge-ok {{
  background: {badge_ok_bg};
  color: {badge_ok_text};
  border: 1px solid rgba(16, 185, 129, 0.3);
  padding: 4px 12px;
  border-radius: 9999px;
  font-size: 0.76rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}}
.badge-warn {{
  background: {badge_warn_bg};
  color: {badge_warn_text};
  border: 1px solid rgba(245, 158, 11, 0.3);
  padding: 4px 12px;
  border-radius: 9999px;
  font-size: 0.76rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}}
.badge-err {{
  background: {badge_err_bg};
  color: {badge_err_text};
  border: 1px solid rgba(239, 68, 68, 0.3);
  padding: 4px 12px;
  border-radius: 9999px;
  font-size: 0.76rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}}
.badge-info {{
  background: {badge_info_bg};
  color: {badge_info_text};
  border: 1px solid rgba(56, 189, 248, 0.3);
  padding: 4px 12px;
  border-radius: 9999px;
  font-size: 0.76rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}}

/* ── Notice & Limitation Callouts ─────────────────────────────────────────── */
.disclaimer {{
  background: {disclaimer_bg};
  border-left: 4px solid {disclaimer_border};
  padding: 12px 18px;
  border-radius: 10px;
  font-size: 0.85rem;
  line-height: 1.5;
  color: {disclaimer_text};
  margin-top: 14px;
}}
.info-notice {{
  background: {bg_card};
  border: 1px solid {border_color};
  border-left: 4px solid #38BDF8;
  padding: 12px 18px;
  border-radius: 10px;
  font-size: 0.86rem;
  line-height: 1.5;
  color: {text_main};
  margin: 10px 0;
  box-shadow: {card_shadow};
}}
.limitation-alert {{
  background: {bg_card};
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-left: 5px solid #EF4444;
  border-radius: 10px;
  padding: 14px 18px;
  font-size: 0.88rem;
  line-height: 1.5;
  color: {text_main};
  margin-bottom: 20px;
  box-shadow: {card_shadow};
}}
.security-note {{
  background: {bg_card};
  border: 1px solid {border_color};
  border-left: 5px solid {accent_green};
  border-radius: 10px;
  padding: 14px 18px;
  font-size: 0.88rem;
  line-height: 1.5;
  color: {text_main};
  margin-bottom: 20px;
  box-shadow: {card_shadow};
}}

/* ── Streamlit Native Metrics ─────────────────────────────────────────────── */
[data-testid="stMetricValue"] {{
  color: {accent_green} !important;
  font-weight: 800 !important;
  letter-spacing: -0.02em !important;
}}
[data-testid="stMetricLabel"] {{
  color: {text_muted} !important;
  font-size: 0.8rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.04em !important;
}}

/* ── Buttons & Download Actions (Tactile Apple-Style Feedback) ─────────────── */
.stButton > button,
div[data-testid="stDownloadButton"] > button,
button[data-testid="baseButton-secondary"],
button[data-testid="baseButton-primary"] {{
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  padding: 0.55rem 1.35rem !important;
  transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1) !important;
}}
.stButton > button:active,
div[data-testid="stDownloadButton"] > button:active,
button:active {{
  transform: scale(0.98) !important;
}}

/* Primary Buttons */
.stButton > button[kind="primary"],
div[data-testid="stDownloadButton"] > button[kind="primary"],
button[data-testid="baseButton-primary"] {{
  background: linear-gradient(135deg, {accent_green} 0%, {accent_green_hover} 100%) !important;
  border: 1px solid {accent_green} !important;
  color: #FFFFFF !important;
  box-shadow: 0 2px 10px {accent_glow} !important;
}}
.stButton > button[kind="primary"]:hover,
div[data-testid="stDownloadButton"] > button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {{
  filter: brightness(1.08) !important;
  box-shadow: 0 4px 16px rgba(16, 185, 129, 0.4) !important;
  color: #FFFFFF !important;
}}

/* Secondary / Standard Buttons */
.stButton > button:not([kind="primary"]),
div[data-testid="stDownloadButton"] > button:not([kind="primary"]),
button[data-testid="baseButton-secondary"],
button:not([kind="primary"]) {{
  background-color: {bg_card} !important;
  border: 1px solid {border_color} !important;
  color: {text_main} !important;
  box-shadow: {card_shadow} !important;
}}
.stButton > button:not([kind="primary"]):hover,
div[data-testid="stDownloadButton"] > button:not([kind="primary"]):hover,
button[data-testid="baseButton-secondary"]:hover,
button:not([kind="primary"]):hover {{
  background-color: {bg_card_hover} !important;
  border-color: {accent_green} !important;
  color: {accent_green} !important;
}}

/* ── Streamlit Form Controls (Inputs, Selectboxes, Dropdowns) ──────────────── */
.stApp input,
.stApp textarea,
.stApp [data-baseweb="input"],
.stApp [data-baseweb="base-input"] {{
  background-color: {bg_card} !important;
  color: {text_main} !important;
  border-color: {border_color} !important;
  border-radius: 9px !important;
}}
.stApp input:focus,
.stApp textarea:focus,
.stApp [data-baseweb="input"]:focus-within {{
  border-color: {accent_green} !important;
  box-shadow: 0 0 0 3px {accent_glow} !important;
}}
.stApp [data-baseweb="select"] > div {{
  background-color: {bg_card} !important;
  color: {text_main} !important;
  border-color: {border_color} !important;
  border-radius: 9px !important;
}}
.stApp [data-baseweb="popover"],
.stApp [data-baseweb="menu"] {{
  background-color: {bg_card} !important;
  color: {text_main} !important;
  border: 1px solid {border_color} !important;
  border-radius: 10px !important;
  box-shadow: {card_shadow} !important;
}}
.stApp [data-baseweb="menu"] li {{
  background-color: {bg_card} !important;
  color: {text_main} !important;
  transition: background 0.1s ease;
}}
.stApp [data-baseweb="menu"] li:hover {{
  background-color: {radio_hover} !important;
  color: {accent_green} !important;
}}
.stApp div[data-testid="stExpander"] {{
  background-color: {bg_card} !important;
  border: 1px solid {border_color} !important;
  border-radius: 12px !important;
  box-shadow: {card_shadow} !important;
}}
.stApp div[data-testid="stExpander"] summary {{
  color: {text_main} !important;
  font-weight: 600 !important;
}}
.stApp .stAlert {{
  background-color: {bg_card} !important;
  color: {text_main} !important;
  border-radius: 10px !important;
  border: 1px solid {border_color} !important;
}}

/* ── Streamlit Tabs (Segmented Control Pill Styling) ───────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
  gap: 8px;
  background-color: transparent !important;
  border-bottom: 1px solid {border_color} !important;
  padding-bottom: 4px;
}}
.stTabs [data-baseweb="tab"] {{
  border-radius: 8px 8px 0 0 !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  padding: 8px 16px !important;
  color: {text_muted} !important;
  border: none !important;
  background: transparent !important;
  transition: all 0.15s ease !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
  color: {text_main} !important;
}}
.stTabs [aria-selected="true"] {{
  color: {accent_green} !important;
  border-bottom: 2px solid {accent_green} !important;
  background: transparent !important;
}}

/* ── Mobile Responsiveness ─────────────────────────────────────────────────── */
@media (max-width: 768px) {{
  .block-container {{ padding-top: 2.5rem !important; }}
  .result-card .yield-number {{ font-size: 2.6rem; }}
  .hero-banner {{ padding: 24px 20px; }}
  .hero-banner h1 {{ font-size: 1.6rem; }}
}}
</style>
""", unsafe_allow_html=True)


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


@st.cache_data(show_spinner=False)
def load_model_ready_cached() -> pd.DataFrame | None:
    try:
        return load_model_ready_data()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_district_cached() -> pd.DataFrame | None:
    try:
        return load_district_data()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_official_cached() -> pd.DataFrame | None:
    try:
        return load_official_2021_25_data()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_combined_cached() -> pd.DataFrame | None:
    try:
        return load_combined_data()
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


# ── Helper: Unified Plotly Theme ──────────────────────────────────────────────
def style_plotly_chart(fig, height: int | None = None):
    grid_col = "rgba(255, 255, 255, 0.07)" if is_dark else "rgba(0, 0, 0, 0.06)"
    axis_line = "rgba(255, 255, 255, 0.12)" if is_dark else "rgba(0, 0, 0, 0.12)"
    layout_update = dict(
        font=dict(family='-apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif', color=text_main, size=12),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=45, b=25),
    )
    if height:
        layout_update["height"] = height
    fig.update_layout(**layout_update)
    fig.update_xaxes(showgrid=True, gridcolor=grid_col, zeroline=False, linecolor=axis_line)
    fig.update_yaxes(showgrid=True, gridcolor=grid_col, zeroline=False, linecolor=axis_line)
    return fig


# ── Helper: page header ───────────────────────────────────────────────────────
def render_page_header(icon: str, title: str, subtitle: str = "", breadcrumb: str = "CropZen"):
    if breadcrumb:
        st.markdown(f'<div class="page-breadcrumb"><span style="color:{accent_green};font-size:0.75rem;">●</span> {breadcrumb}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">{icon} {title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Sleek brand lockup (Custom SVG leaf/sprout icon in emerald glass squircle)
    st.markdown(f"""
    <div style="padding: 10px 4px 6px 4px; display: flex; align-items: center; gap: 12px;">
      <div style="width: 38px; height: 38px; border-radius: 10px; background: linear-gradient(135deg, {accent_green} 0%, {accent_green_hover} 100%); display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px {accent_glow}; flex-shrink: 0;">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"></path>
          <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"></path>
        </svg>
      </div>
      <div>
        <div style="display: flex; align-items: center; gap: 8px;">
          <span style="font-size: 1.35rem; font-weight: 800; letter-spacing: -0.025em; color: {text_main};">CropZen</span>
          <span style="font-size: 0.65rem; font-weight: 700; padding: 2px 7px; border-radius: 9999px; background: {badge_ok_bg}; color: {badge_ok_text}; border: 1px solid rgba(16, 185, 129, 0.3); letter-spacing: 0.04em;">v2.4 PRO</span>
        </div>
        <div style="font-size: 0.72rem; color: {text_muted}; margin-top: 2px; font-weight: 500;">AI Precision Agriculture</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<hr style="border:none;border-top:1px solid {border_color};margin:10px 0 14px 0;" />', unsafe_allow_html=True)

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

    st.markdown(f'<hr style="border:none;border-top:1px solid {border_color};margin:12px 0;" />', unsafe_allow_html=True)

    # Theme selector (Light / Dark mode)
    theme_choice = st.radio(
        "🎨 " + T("theme", lang),
        options=["light", "dark"],
        format_func=lambda t: "☀️ " + T("theme_light", lang) if t == "light" else "🌙 " + T("theme_dark", lang),
        index=0 if st.session_state.theme == "light" else 1,
        horizontal=True,
        key="theme_selector",
    )
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

    st.markdown(f'<hr style="border:none;border-top:1px solid {border_color};margin:12px 0;" />', unsafe_allow_html=True)

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

    # Handle programmatic navigation override seamlessly
    nav_index = 0
    if "_nav_override" in st.session_state:
        target_nav = st.session_state.pop("_nav_override")
        for idx, (p_label, p_code) in enumerate(pages.items()):
            if p_code == target_nav:
                nav_index = idx
                break

    selected = st.radio(
        "Navigation",
        options=list(pages.keys()),
        index=nav_index,
        label_visibility="collapsed",
    )
    page = pages[selected]

    st.markdown(f'<hr style="border:none;border-top:1px solid {border_color};margin:14px 0 10px 0;" />', unsafe_allow_html=True)

    # Model status card
    if is_model_ready():
        eval_meta = load_eval_cached()
        best = eval_meta.get("best_model", "Unknown") if eval_meta else "Unknown"
        metrics = eval_meta.get("metrics", {}).get(best, {}) if eval_meta else {}
        r2_val = metrics.get("R2", 0)
        mae_val = metrics.get("MAE", 0)
        st.markdown(f"""
        <div style="background:{bg_card};border:1px solid {border_color};border-radius:10px;padding:12px 14px;box-shadow:{card_shadow};">
          <div style="display:flex;align-items:center;gap:8px;font-size:0.74rem;font-weight:700;color:{badge_ok_text};">
            <span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{accent_green};box-shadow:0 0 6px {accent_green};"></span>
            PRODUCTION MODEL ACTIVE
          </div>
          <div style="font-size:0.88rem;font-weight:700;color:{text_main};margin-top:3px;">{best} Regressor</div>
          <div style="font-size:0.72rem;color:{text_muted};margin-top:3px;">
            R² <strong>{r2_val:.4f}</strong> &bull; MAE <strong>&plusmn;{mae_val:.2f} t/ha</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge-warn">⚠️ Model Not Loaded</span>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size:0.72rem;color:{text_subdued};margin-top:16px;text-align:center;">
      5th Sem CSE Design Project &bull; Precision Agriculture
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# A. HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "home":
    # Live production metrics for hero
    eval_meta = load_eval_cached()
    best_mname = eval_meta.get("best_model", "LightGBM") if eval_meta else "LightGBM"
    m_info = eval_meta.get("metrics", {}).get(best_mname, {}) if eval_meta else {}
    r2_hero = m_info.get("R2", 0.9774)
    mae_hero = m_info.get("MAE", 8.22)

    # Hero banner
    st.markdown(f"""
    <div class="hero-banner">
      <div class="hero-badge" style="margin-bottom:16px;">
        <span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#34D399;box-shadow:0 0 8px #34D399;"></span>
        Production Model Active &bull; {best_mname} (R² {r2_hero:.4f} &bull; MAE &plusmn;{mae_hero:.2f} t/ha)
      </div>
      <h1>🌾 Precision Agricultural Intelligence &amp; Yield Estimation</h1>
      <p>{T('home_description', lang)} Evaluated objectively across 17,983 multi-decade Indian records with standardized coconut units and leakage-free feature pipelines.</p>
      <div class="hero-badge-row">
        <div class="hero-badge">📊 17,983 Training Records (2000–2020)</div>
        <div class="hero-badge">🌱 55 Crops Across 30 States</div>
        <div class="hero-badge">🎯 {r2_hero:.4f} R² Test Accuracy</div>
        <div class="hero-badge">🥥 Official CDB Coconut Standardization</div>
      </div>
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
    df = load_model_ready_cached()
    if df is None:
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
            df_off = load_official_cached()
            max_yr = 2025 if df_off is not None else (int(df['Crop_Year'].max()) if "Crop_Year" in df.columns else 2020)
            min_yr = int(df['Crop_Year'].min()) if "Crop_Year" in df.columns else 2000
            year_range = f"{min_yr}–{max_yr}"
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-value" style="font-size:1.3rem;">{year_range}</div>
              <div class="stat-label">{T('home_years', lang)} (incl. 2024–25)</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Multi-Dataset Registry on Home ─────────────────────────────────────
        st.markdown('<div class="section-title">Verified Data Assets & Usage Scope</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.88rem;color:{text_muted};margin-bottom:12px;">
          CropZen maintains four dedicated agricultural data assets, cleanly separated between <strong>machine-learning training</strong> and <strong>exploratory / official reporting</strong>.
        </div>
        """, unsafe_allow_html=True)

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.markdown(f"""
            <div class="card" style="padding:14px;border-top:3px solid {accent_green};">
              <div style="font-size:0.75rem;font-weight:700;color:{accent_green};">MODEL TRAINING</div>
              <div style="font-weight:700;font-size:0.95rem;margin:4px 0;color:{text_main};">Model-Ready (2000–20)</div>
              <div style="font-size:0.78rem;color:{text_muted};">17,983 records · State level<br>Complete features (Area, Rainfall, Fertilizer, Pesticide) · Evaluated on held-out test split.</div>
            </div>
            """, unsafe_allow_html=True)
        with d2:
            st.markdown(f"""
            <div class="card" style="padding:14px;border-top:3px solid #0284C7;">
              <div style="font-size:0.75rem;font-weight:700;color:#0284C7;">ORIGINAL BASELINE</div>
              <div style="font-weight:700;font-size:0.95rem;margin:4px 0;color:{text_main};">Historical (1997–20)</div>
              <div style="font-size:0.78rem;color:{text_muted};">19,689 records · State level<br>30 states, 55 crops. Historical baseline benchmark for multi-model comparisons.</div>
            </div>
            """, unsafe_allow_html=True)
        with d3:
            st.markdown(f"""
            <div class="card" style="padding:14px;border-top:3px solid #D97706;">
              <div style="font-size:0.75rem;font-weight:700;color:#D97706;">DISTRICT EXPLORATION</div>
              <div style="font-weight:700;font-size:0.95rem;margin:4px 0;color:{text_main};">District Level (2000–20)</div>
              <div style="font-size:0.78rem;color:{text_muted};">311,488 records · 706 districts<br>Granular local production. Missing rainfall/inputs — used for geospatial EDA, not model training.</div>
            </div>
            """, unsafe_allow_html=True)
        with d4:
            st.markdown(f"""
            <div class="card" style="padding:14px;border-top:3px solid #9333EA;">
              <div style="font-size:0.75rem;font-weight:700;color:#9333EA;">RECENT OFFICIAL</div>
              <div style="font-weight:700;font-size:0.95rem;margin:4px 0;color:{text_main};">All-India (2020–25)</div>
              <div style="font-size:0.78rem;color:{text_muted};">225 records · 2020-21 to 2024-25<br>Official production aggregates. Observed Yield missing — supervised validation prohibited.</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Model status inline card ───────────────────────────────────────────
        st.markdown("")
        if is_model_ready():
            eval_meta = load_eval_cached()
            best = eval_meta.get("best_model", "Unknown") if eval_meta else "Unknown"
            dataset_name = eval_meta.get("dataset_label", "CropZen Dataset") if eval_meta else "CropZen Dataset"
            metrics = eval_meta.get("metrics", {}).get(best, {}) if eval_meta else {}
            r2  = metrics.get("R2",   0)
            mae = metrics.get("MAE",  0)
            rmse= metrics.get("RMSE", 0)
            st.markdown(f"""
            <div class="card card-accent" style="margin-top:8px;">
              <div style="font-size:0.85rem;color:{text_muted};margin-bottom:6px;">Active Production Model &bull; {dataset_name}</div>
              <div style="font-size:1.1rem;font-weight:700;color:{accent_green};margin-bottom:8px;">⭐ {best}</div>
              <div style="display:flex;gap:32px;flex-wrap:wrap;">
                <div><div style="font-size:1.4rem;font-weight:700;color:{accent_green};">{r2:.4f}</div><div style="font-size:0.75rem;color:{text_muted};">R² (Held-Out Test)</div></div>
                <div><div style="font-size:1.4rem;font-weight:700;color:{text_main};">{mae:.2f}</div><div style="font-size:0.75rem;color:{text_muted};">MAE (t/ha)</div></div>
                <div><div style="font-size:1.4rem;font-weight:700;color:{text_main};">{rmse:.2f}</div><div style="font-size:0.75rem;color:{text_muted};">RMSE (t/ha)</div></div>
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
                template=plotly_template,
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              margin=dict(l=0, r=0, t=40, b=0),
                              font=dict(color=text_main),
                              plot_bgcolor=plotly_bg, paper_bgcolor=plotly_bg)
            st.plotly_chart(style_plotly_chart(fig), use_container_width=True)

        with col_b:
            season_counts = df["Season"].value_counts().reset_index()
            season_counts.columns = ["Season", "Count"]
            fig2 = px.pie(
                season_counts, names="Season", values="Count",
                title="Distribution of Records by Season",
                color_discrete_sequence=["#1F7A4D", "#2D9D63", "#4DB87E", "#80CEA0", "#B3E2C7", "#E0F4EC"],
                template=plotly_template,
            )
            fig2.update_layout(margin=dict(l=0, r=0, t=40, b=0),
                               font=dict(color=text_main),
                               paper_bgcolor=plotly_bg)
            st.plotly_chart(style_plotly_chart(fig2), use_container_width=True)
    else:
        st.error("Dataset not found. Check data/crop_yield.csv.")

    # ── How it works & Interactive Pipeline ────────────────────────────────────
    st.markdown('<div class="section-title">End-to-End Prediction Pipeline Architecture</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="pipeline-grid">
      <div class="pipeline-step">
        <div class="pipeline-step-header">
          <div class="pipeline-step-num">1</div>
          <div class="pipeline-step-title">Agronomic Inputs</div>
        </div>
        <div class="pipeline-step-desc">Captures crop variety, state, seasonal calendar, cultivated area, precipitation, and fertilizer/pesticide levels.</div>
      </div>
      <div class="pipeline-step">
        <div class="pipeline-step-header">
          <div class="pipeline-step-num">2</div>
          <div class="pipeline-step-title">Leakage Isolation</div>
        </div>
        <div class="pipeline-step-desc">Strictly excludes post-harvest <code>Production</code> to prevent leakage. Applies ordinal encoding with unseen category safety.</div>
      </div>
      <div class="pipeline-step">
        <div class="pipeline-step-header">
          <div class="pipeline-step-num">3</div>
          <div class="pipeline-step-title">Gradient Boosting</div>
        </div>
        <div class="pipeline-step-desc">Evaluates leaf-wise tree ensembles (LightGBM/XGBoost) optimizing non-linear feature interactions and regional yield curves.</div>
      </div>
      <div class="pipeline-step">
        <div class="pipeline-step-header">
          <div class="pipeline-step-num">4</div>
          <div class="pipeline-step-title">Empirical Bounds</div>
        </div>
        <div class="pipeline-step-desc">Returns yield in native statistical units (&plusmn;MAE confidence bounds) with official CDB coconut mass conversion (1.4 kg/nut).</div>
      </div>
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
        st.components.v1.html(get_voice_input_html(lang_code, is_dark=is_dark), height=270, scrolling=False)
        st.caption("🔒 *Note: Requires microphone permission in your browser (Google Chrome or Microsoft Edge recommended).*")

        transcript = st.text_input(
            "📝 " + T("predict_recognized", lang) + " (Paste transcript here or type in Telugu, Hindi, or English):",
            value=st.session_state.voice_transcript,
            key="voice_text_input",
            placeholder="Click 'Copy Text' above and paste here...",
        )
        if transcript:
            st.session_state.voice_transcript = transcript
            parsed_voice = parse_multilingual_voice_text(transcript)
            det_lang = parsed_voice["detected_lang"]
            st.session_state.user_spoken_lang = det_lang
            lang_label = "🇮🇳 తెలుగు (Telugu)" if det_lang == "te" else ("🇮🇳 हिन्दी (Hindi)" if det_lang == "hi" else "🇬🇧 English")

            col_v1, col_v2 = st.columns([3, 1])
            with col_v1:
                entities_str = ", ".join(parsed_voice["entities_found"]) if parsed_voice["entities_found"] else "No specific entities recognized yet."
                st.info(f"🗣️ **Detected Voice Language:** {lang_label}\n\n**Extracted Parameters:** {entities_str}")
            with col_v2:
                if st.button("⚡ Auto-Fill Form", use_container_width=True, type="primary"):
                    crops_list = schema["categories"].get("Crop", [])
                    states_list = schema["categories"].get("State", [])
                    seasons_list = schema["categories"].get("Season", [])
                    if parsed_voice["crop"] and parsed_voice["crop"] in crops_list:
                        st.session_state["sel_crop"] = parsed_voice["crop"]
                    if parsed_voice["state"] and parsed_voice["state"] in states_list:
                        st.session_state["sel_state"] = parsed_voice["state"]
                    if parsed_voice["season"] and parsed_voice["season"] in seasons_list:
                        st.session_state["sel_season"] = parsed_voice["season"]
                    if parsed_voice["year"]:
                        st.session_state["sel_year"] = int(parsed_voice["year"])
                    if parsed_voice["area"]:
                        st.session_state["sel_area"] = float(parsed_voice["area"])
                    if parsed_voice["rainfall"]:
                        st.session_state["sel_rainfall"] = float(parsed_voice["rainfall"])
                    if parsed_voice["fertilizer"]:
                        st.session_state["sel_fertilizer"] = float(parsed_voice["fertilizer"])
                    if parsed_voice["pesticide"]:
                        st.session_state["sel_pesticide"] = float(parsed_voice["pesticide"])
                    st.success("✅ Form inputs populated from your spoken words!")
                    st.rerun()

    st.markdown("---")

    # ── Prediction form ───────────────────────────────────────────────────────
    with st.form("prediction_form"):
        # Step 1: Crop & Location
        st.markdown('<div class="section-title">Step 1 — Crop & Location</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            crops = schema["categories"].get("Crop", [])
            if "sel_crop" not in st.session_state or st.session_state["sel_crop"] not in crops:
                st.session_state["sel_crop"] = "Rice" if "Rice" in crops else (crops[0] if crops else "")
            crop_val = st.selectbox(
                T("predict_crop", lang),
                options=crops, key="sel_crop",
                help="Select the crop variety from the 55 supported crops.",
            )
        with col2:
            states = schema["categories"].get("State", [])
            if "sel_state" not in st.session_state or st.session_state["sel_state"] not in states:
                st.session_state["sel_state"] = "Andhra Pradesh" if "Andhra Pradesh" in states else (states[0] if states else "")
            state_val = st.selectbox(
                T("predict_state", lang),
                options=states, key="sel_state",
                help="Select the Indian state where the crop was/will be cultivated.",
            )

        seasons = schema["categories"].get("Season", [])
        if "sel_season" not in st.session_state or st.session_state["sel_season"] not in seasons:
            st.session_state["sel_season"] = "Kharif" if "Kharif" in seasons else (seasons[0] if seasons else "")
        season_val = st.selectbox(
            T("predict_season", lang),
            options=seasons, key="sel_season",
            help="Agricultural season: Kharif (June–Oct), Rabi (Oct–Mar), Whole Year, Summer, Autumn, or Winter.",
        )

        # Step 2: Time & Land
        st.markdown('<div class="section-title">Step 2 — Time & Land</div>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            if "sel_year" not in st.session_state:
                st.session_state["sel_year"] = 2019
            year_val = st.number_input(
                T("predict_year", lang),
                min_value=1990, max_value=2030,
                step=1, key="sel_year",
                help="Crop year (training data spans 1997–2020; 5th–95th percentile: 1998–2019).",
            )
        with col4:
            if "sel_area" not in st.session_state:
                st.session_state["sel_area"] = 5000.0
            area_val = st.number_input(
                T("predict_area", lang) + " (ha)",
                min_value=0.001, max_value=1e8,
                key="sel_area",
                help="Cultivated area in hectares. Typical range: 100 – 500,000 ha.",
                format="%.2f",
            )

        # Step 3: Environment & Inputs
        st.markdown('<div class="section-title">Step 3 — Environment & Agricultural Inputs</div>', unsafe_allow_html=True)
        col5, col6, col7 = st.columns(3)
        with col5:
            if "sel_rainfall" not in st.session_state:
                st.session_state["sel_rainfall"] = 1200.0
            rainfall_val = st.number_input(
                T("predict_rainfall", lang) + " (mm)",
                min_value=0.0, max_value=10000.0,
                key="sel_rainfall",
                help="Annual rainfall in mm. Typical range: 200 – 3,000 mm.",
                format="%.1f",
            )
        with col6:
            if "sel_fertilizer" not in st.session_state:
                st.session_state["sel_fertilizer"] = 500000.0
            fertilizer_val = st.number_input(
                T("predict_fertilizer", lang) + " (kg)",
                min_value=0.0, max_value=1e11,
                key="sel_fertilizer",
                help="Total fertilizer used in kg. Typical range: 1,000 – 10,000,000 kg.",
                format="%.0f",
            )
        with col7:
            if "sel_pesticide" not in st.session_state:
                st.session_state["sel_pesticide"] = 1000.0
            pesticide_val = st.number_input(
                T("predict_pesticide", lang) + " (kg)",
                min_value=0.0, max_value=1e8,
                key="sel_pesticide",
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
                target_unit = result.get("target_unit", "tonnes/ha")
                is_coconut = (target_unit == "nuts/ha")
                converted_t_per_ha = result.get("converted_yield_tonnes", pred_yield)

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

                # ── Determine user's spoken or active language for response ────
                answer_lang = st.session_state.get("user_spoken_lang", lang)
                if answer_lang not in ("te", "hi", "en"):
                    answer_lang = lang

                res_card = build_multilingual_result_card(
                    crop=crop_val,
                    predicted_yield=pred_yield,
                    model_name=model_name,
                    area=area_val,
                    lang=answer_lang,
                )

                # Localized banner in the language the user spoke (Telugu or Hindi)
                if answer_lang in ("te", "hi"):
                    spoken_bg = "#064E3B" if is_dark else "#F2F8F4"
                    spoken_border = "#22C55E" if is_dark else "#1F7A4D"
                    spoken_title_color = "#6EE7B7" if is_dark else "#14532D"
                    st.markdown(f"""
                    <div style="background:{spoken_bg};border-left:5px solid {spoken_border};border-radius:10px;padding:16px 20px;margin-bottom:16px;box-shadow:0 1px 4px rgba(0,0,0,0.06);">
                      <div style="font-size:1.05rem;font-weight:700;color:{spoken_title_color};margin-bottom:6px;">{res_card['title']}</div>
                      <div style="font-size:0.95rem;color:{text_main};margin-bottom:4px;">{res_card['headline']}</div>
                      <div style="font-size:1.15rem;font-weight:700;color:{accent_green};margin-bottom:4px;">{res_card['yield_text']}</div>
                      <div style="font-size:0.92rem;color:{text_main};margin-bottom:6px;">{res_card['production_text']}</div>
                      <div style="font-size:0.8rem;color:{text_muted};">{res_card['note']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                if is_coconut:
                    unit_label = "nuts per hectare (nuts/ha)"
                    unit_pill = f'<span class="badge-ok">🥥 Official CDB Standard Unit</span>'
                    prod_context = (
                        f"Estimated harvest for {area_val:,.1f} ha: <strong>{pred_yield * area_val:,.0f} nuts</strong> "
                        f"(&asymp; <strong>{converted_t_per_ha * area_val:,.1f} metric tonnes</strong> fresh mass)"
                    )
                    conversion_box = f"""
                    <div style="margin-top:14px;padding:12px 16px;background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.25);border-radius:10px;font-size:0.85rem;color:{text_main};text-align:left;">
                      🥥 <strong>CDB Statistical Unit Conversion:</strong> Native prediction: <strong>{pred_yield:,.1f} nuts/ha</strong>. Standard mass equivalence: <strong>{converted_t_per_ha:.2f} t/ha</strong> based on national agronomic benchmark (1.4 kg whole fresh nut weight).
                    </div>
                    """
                else:
                    unit_label = f"metric tonnes per hectare (t/ha)"
                    unit_pill = f'<span class="badge-info">🌾 Standard Food Crop Metric</span>'
                    prod_context = f"Estimated total production for {area_val:,.1f} ha: <strong>{pred_yield * area_val:,.1f} metric tonnes</strong>"
                    conversion_box = ""

                st.markdown(f"""
                <div class="result-card">
                  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:14px;">
                    <div style="display:flex;align-items:center;gap:8px;">
                      {dist_badge}
                      {unit_pill}
                    </div>
                    <span style="font-size:0.78rem;color:{text_muted};font-weight:600;letter-spacing:0.04em;text-transform:uppercase;">ML Yield Estimate</span>
                  </div>
                  <div style="font-size:1.05rem;font-weight:700;color:{text_main};margin-bottom:6px;">
                    <span style="color:{accent_green};">🌾 {crop_val}</span> &bull; {season_val} &bull; {state_val} &bull; {year_val}
                  </div>
                  <div class="yield-number">{pred_yield:,.2f}</div>
                  <div class="yield-unit">{unit_label}</div>
                  <div class="result-context">{prod_context}</div>
                  {conversion_box}
                  <div class="result-meta">
                    <div class="result-meta-item"><strong>Model:</strong> {model_name}</div>
                    <div class="result-meta-item"><strong>Held-Out MAE:</strong> &plusmn;{mae_display} {target_unit}</div>
                    <div class="result-meta-item"><strong>Test R²:</strong> {r2_display}</div>
                    <div class="result-meta-item"><strong>Cultivated Area:</strong> {area_val:,.1f} ha</div>
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

                # ── TTS button in user's spoken language ────────────────────────
                speech_text, tts_lang_code = build_multilingual_spoken_answer(
                    crop=crop_val,
                    predicted_yield=pred_yield,
                    model_name=model_name,
                    area=area_val,
                    lang=answer_lang,
                )
                st.markdown(f"**{res_card['listen_label']}**")
                st.components.v1.html(
                    get_tts_html(speech_text, tts_lang_code, is_dark=is_dark),
                    height=105, scrolling=False,
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
                            template=plotly_template,
                            title=f"Feature Importance — {model_name}",
                        )
                        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                                          margin=dict(l=0, r=0, t=40, b=0),
                                          font=dict(color=text_main),
                                          plot_bgcolor=plotly_bg, paper_bgcolor=plotly_bg)
                        st.plotly_chart(style_plotly_chart(fig), use_container_width=True)
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
                st.caption("**Predicted Target:** `Predicted_Yield` in native statistical unit (`tonnes/ha` for food crops; `nuts/ha` for Coconut). `Converted_Yield_Tonnes` shows mass equivalence.")
                display_cols = BATCH_REQUIRED_COLUMNS + ["Predicted_Yield", "Yield_Unit", "Converted_Yield_Tonnes"]
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
                        template=plotly_template, nbins=30,
                    )
                    fig.update_layout(font=dict(color=text_main), plot_bgcolor=plotly_bg, paper_bgcolor=plotly_bg)
                    st.plotly_chart(style_plotly_chart(fig), use_container_width=True)

                with col_chart2:
                    if result_df.loc[ok_mask, "Crop"].nunique() > 1:
                        crop_avg = result_df.loc[ok_mask].groupby("Crop")["Predicted_Yield"].mean().reset_index()
                        fig_crop = px.bar(
                            crop_avg, x="Crop", y="Predicted_Yield",
                            title="Average Predicted Yield by Crop (t/ha)",
                            labels={"Predicted_Yield": "Avg Yield (t/ha)"},
                            color="Crop", template=plotly_template,
                        )
                        fig_crop.update_layout(font=dict(color=text_main), plot_bgcolor=plotly_bg, paper_bgcolor=plotly_bg)
                        st.plotly_chart(style_plotly_chart(fig_crop), use_container_width=True)
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
        subtitle="Multi-Dataset Explorer — inspect state-level model data, district historicals, official all-India statistics, and combined assets.",
        breadcrumb="CropZen › Dataset Explorer",
    )

    _chart_layout = dict(font=dict(color=text_main), plot_bgcolor=plotly_bg, paper_bgcolor=plotly_bg)

    ex_tab1, ex_tab2, ex_tab3, ex_tab4 = st.tabs([
        "🌾 Model-Ready State Data (2000–20)",
        "🗺️ District Historicals (2000–20)",
        "🏛️ Official All-India (2020–25)",
        "📂 Combined Asset (2000–25)"
    ])

    # ── TAB 1: MODEL-READY HISTORICAL DATA ────────────────────────────────────
    with ex_tab1:
        st.markdown("""
        <div class="info-notice">
          <strong>Dataset:</strong> <code>cropzen_model_ready_2000_2020.csv</code> (State-level, 17,983 records)<br>
          <strong>Scope:</strong> Full feature set (Crop, State, Season, Year, Area, Rainfall, Fertilizer, Pesticide, Yield). Used to train and test the predictive ML models.
        </div>
        """, unsafe_allow_html=True)

        df_mr = load_model_ready_cached()
        if df_mr is None:
            df_mr = load_dataset()

        if df_mr is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Records", f"{len(df_mr):,}")
            c2.metric("Unique Crops", df_mr["Crop"].nunique())
            c3.metric("States", df_mr["State"].nunique())
            c4.metric("Years", f"{int(df_mr['Crop_Year'].min())}–{int(df_mr['Crop_Year'].max())}")

            # Top Crops & Yield distributions
            col_a, col_b = st.columns(2)
            with col_a:
                fig_mr1 = px.histogram(
                    df_mr, x="Yield", nbins=50,
                    title="Yield Distribution (t/ha)",
                    labels={"Yield": "Yield (t/ha)"},
                    color_discrete_sequence=["#1F7A4D"],
                    template=plotly_template,
                )
                fig_mr1.update_layout(**_chart_layout)
                st.plotly_chart(style_plotly_chart(fig_mr1), use_container_width=True)

            with col_b:
                top_crops_mr = df_mr.groupby("Crop")["Yield"].median().nlargest(12).reset_index()
                fig_mr2 = px.bar(
                    top_crops_mr, x="Yield", y="Crop", orientation="h",
                    title="Top 12 Crops by Median Yield (t/ha)",
                    labels={"Yield": "Median Yield (t/ha)"},
                    color="Yield", color_continuous_scale=["#C9E8D5", "#1F7A4D"],
                    template=plotly_template,
                )
                fig_mr2.update_layout(coloraxis_showscale=False, **_chart_layout)
                st.plotly_chart(style_plotly_chart(fig_mr2), use_container_width=True)

            with st.expander("📋 View Sample Records (first 25 rows)"):
                st.dataframe(df_mr.head(25), use_container_width=True)
        else:
            st.error("Model-ready dataset unavailable.")

    # ── TAB 2: DISTRICT HISTORICALS ───────────────────────────────────────────
    with ex_tab2:
        st.markdown("""
        <div class="info-notice">
          <strong>Dataset:</strong> <code>cropzen_india_district_2000_2020.csv</code> (District-level, 311,488 records)<br>
          <strong>Scope:</strong> High-resolution spatial data covering <strong>706 districts</strong> across 37 states.<br>
          <strong>Notice:</strong> <code>Annual_Rainfall</code>, <code>Fertilizer</code>, and <code>Pesticide</code> are 100% missing in this dataset. It is used exclusively for localized geographic exploration, NOT for model training.
        </div>
        """, unsafe_allow_html=True)

        df_dist = load_district_cached()
        if df_dist is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("District Records", f"{len(df_dist):,}")
            c2.metric("Districts", f"{df_dist['District'].nunique():,}")
            c3.metric("States", df_dist["State"].nunique())
            c4.metric("Avg Area/Record", f"{df_dist['Area'].mean():,.1f} ha")

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                top_dist_prod = df_dist.groupby("State")["Production"].sum().nlargest(10).reset_index()
                fig_d1 = px.bar(
                    top_dist_prod, x="Production", y="State", orientation="h",
                    title="Top 10 States by Total Production (District Sum)",
                    color="Production", color_continuous_scale=["#FDE68A", "#D97706"],
                    template=plotly_template,
                )
                fig_d1.update_layout(coloraxis_showscale=False, **_chart_layout)
                st.plotly_chart(style_plotly_chart(fig_d1), use_container_width=True)

            with col_d2:
                top_dist_crops = df_dist["Crop"].value_counts().head(10).reset_index()
                top_dist_crops.columns = ["Crop", "Count"]
                fig_d2 = px.pie(
                    top_dist_crops, names="Crop", values="Count",
                    title="Top 10 Most Frequently Reported Crops Across Districts",
                    template=plotly_template,
                )
                fig_d2.update_layout(**_chart_layout)
                st.plotly_chart(style_plotly_chart(fig_d2), use_container_width=True)

            with st.expander("📋 View Sample District Records (first 25 rows)"):
                st.dataframe(df_dist[["State", "District", "Crop", "Crop_Year", "Season", "Area", "Production", "Yield"]].head(25), use_container_width=True)
        else:
            st.error("District-level dataset unavailable.")

    # ── TAB 3: OFFICIAL ALL-INDIA 2021-2025 ────────────────────────────────────
    with ex_tab3:
        st.markdown("""
        <div class="limitation-alert">
          <strong>⚠️ Official 2020-21 to 2024-25 Validation Notice:</strong><br>
          This official Ministry dataset (<code>cropzen_india_official_all_india_2021_2025.csv</code>, 225 rows) provides aggregate national crop production statistics.<br>
          <strong>Critical Validation Status:</strong> The <code>Yield</code> and <code>Area</code> columns are <strong>100% missing</strong>. Supervised prediction accuracy CANNOT be validated against these records. We report official observed production statistics below without fabricating missing yield values.
        </div>
        """, unsafe_allow_html=True)

        df_off = load_official_cached()
        if df_off is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Official Records", len(df_off))
            c2.metric("Agricultural Years", "2020-21 to 2024-25")
            c3.metric("Geography", "All-India Aggregate")

            col_o1, col_o2 = st.columns(2)
            with col_o1:
                # Top crops by production in lakh tonnes
                # Filter to lakh tonnes to avoid mixing units with cotton bales
                tonnes_df = df_off[df_off["production_unit"] == "lakh tonnes"].copy()
                top_off_crops = tonnes_df.groupby("Crop")["Production"].mean().nlargest(10).reset_index()
                fig_o1 = px.bar(
                    top_off_crops, x="Production", y="Crop", orientation="h",
                    title="Top 10 Food Crops — Mean National Production (Lakh Tonnes)",
                    labels={"Production": "Production (Lakh Tonnes)"},
                    color="Production", color_continuous_scale=["#E9D5FF", "#7E22CE"],
                    template=plotly_template,
                )
                fig_o1.update_layout(coloraxis_showscale=False, **_chart_layout)
                st.plotly_chart(style_plotly_chart(fig_o1), use_container_width=True)

            with col_o2:
                # Production trend over recent years for major crops
                trend_crops = ["Rice", "Wheat", "Maize"]
                trend_df = df_off[df_off["Crop"].isin(trend_crops) & (df_off["production_unit"] == "lakh tonnes")].groupby(["crop_year_label", "Crop"])["Production"].sum().reset_index()
                fig_o2 = px.line(
                    trend_df, x="crop_year_label", y="Production", color="Crop",
                    title="Production Trajectory (2020-21 to 2024-25)",
                    labels={"crop_year_label": "Agricultural Year", "Production": "Production (Lakh Tonnes)"},
                    markers=True,
                    template=plotly_template,
                )
                fig_o2.update_layout(**_chart_layout)
                st.plotly_chart(style_plotly_chart(fig_o2), use_container_width=True)

            with st.expander("📋 View Complete Official Statistics Table"):
                st.dataframe(df_off[["Crop", "Season", "crop_year_label", "Production", "production_unit", "estimate_status", "source"]], use_container_width=True)
        else:
            st.error("Official 2021-25 dataset unavailable.")

    # ── TAB 4: COMBINED EXPLORATORY ASSET ──────────────────────────────────────
    with ex_tab4:
        st.markdown("""
        <div class="info-notice">
          <strong>Dataset:</strong> <code>cropzen_india_combined_2000_2025.csv</code> (311,713 rows)<br>
          <strong>Scope:</strong> Master unified asset indexing both district records (2000–2020) and all-India official reports (2020–2025).<br>
          <strong>Methodological Warning:</strong> Never aggregate records across different <code>geography_level</code> entries without grouping, as doing so introduces extreme scale distortion.
        </div>
        """, unsafe_allow_html=True)

        df_comb = load_combined_cached()
        if df_comb is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Rows", f"{len(df_comb):,}")
            c2.metric("District Rows", f"{(df_comb['geography_level'] == 'district').sum():,}")
            c3.metric("All-India Rows", f"{(df_comb['geography_level'] == 'all_india').sum():,}")

            st.dataframe(
                df_comb.groupby(["geography_level", "estimate_status"])
                .agg(record_count=("Crop_Year", "count"), min_year=("Crop_Year", "min"), max_year=("Crop_Year", "max"))
                .reset_index(),
                use_container_width=True
            )
        else:
            st.error("Combined dataset unavailable.")


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
    _chart_layout = dict(font=dict(color=text_main), plot_bgcolor=plotly_bg, paper_bgcolor=plotly_bg)

    # ── Best model summary ────────────────────────────────────────────────────
    trained_dataset = eval_meta.get("dataset_label", "CropZen Dataset")
    trained_file = eval_meta.get("dataset_file", "cropzen_model_ready_2000_2020.csv")
    split_method = eval_meta.get("split_method", "Random 80/20")
    st.markdown(f"""
    <div class="card card-accent">
      <div style="font-size:0.8rem;color:{text_muted};margin-bottom:4px;">Best Performing Model &bull; Evaluated on Held-Out Test Set</div>
      <div style="font-size:1.4rem;font-weight:700;color:{accent_green};">⭐ {best_model}</div>
      <div style="font-size:0.85rem;color:{text_main};margin-top:4px;">
        <strong>Training Dataset:</strong> {trained_dataset} (<code>{trained_file}</code>)<br>
        <strong>Split Strategy:</strong> {split_method} &nbsp;|&nbsp;
        <strong>Train Size:</strong> {eval_meta.get('train_size', 0):,} rows &nbsp;|&nbsp;
        <strong>Held-Out Test Size:</strong> {eval_meta.get('test_size', 0):,} rows
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
            template=plotly_template,
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-20, **_chart_layout)
        st.plotly_chart(style_plotly_chart(fig), use_container_width=True)
        st.caption(T("perf_mae_explain", lang))

    with col2:
        fig2 = px.bar(
            metrics_df, x="Model", y="RMSE (t/ha)",
            title=T("perf_rmse", lang),
            color="Model", color_discrete_sequence=_bar_colors,
            template=plotly_template,
        )
        fig2.update_layout(showlegend=False, xaxis_tickangle=-20, **_chart_layout)
        st.plotly_chart(style_plotly_chart(fig2), use_container_width=True)
        st.caption(T("perf_rmse_explain", lang))

    with col3:
        fig3 = px.bar(
            metrics_df, x="Model", y="R²",
            title=T("perf_r2", lang),
            color="Model", color_discrete_sequence=_bar_colors,
            template=plotly_template,
        )
        fig3.update_layout(showlegend=False, xaxis_tickangle=-20, **_chart_layout)
        st.plotly_chart(style_plotly_chart(fig3), use_container_width=True)
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
                template=plotly_template,
            )
            # Perfect-prediction diagonal
            fig4.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines", name="Perfect Prediction (y = x)",
                line=dict(color="#DC2626", dash="dash", width=1.5),
            ))
            fig4.update_layout(**_chart_layout)
            st.plotly_chart(style_plotly_chart(fig4), use_container_width=True)
            st.caption("Points close to the red diagonal line indicate accurate predictions. Clusters far from the line indicate systematic errors.")

        with col_p2:
            scatter_df["Residual"] = scatter_df["Predicted"] - scatter_df["Actual"]
            fig5 = px.scatter(
                scatter_df, x="Predicted", y="Residual",
                title=f"Residuals Plot — {best_model}",
                labels={"Predicted": "Predicted Yield (t/ha)", "Residual": "Residual (Predicted − Actual)"},
                opacity=0.4,
                color_discrete_sequence=["#3B82F6"],
                template=plotly_template,
            )
            fig5.add_hline(y=0, line_dash="dash", line_color="#DC2626", line_width=1.5)
            fig5.update_layout(**_chart_layout)
            st.plotly_chart(style_plotly_chart(fig5), use_container_width=True)
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
                template=plotly_template,
            )
            fig6.update_layout(coloraxis_showscale=False, **_chart_layout)
            st.plotly_chart(style_plotly_chart(fig6), use_container_width=True)
            st.caption(T("perf_feat_note", lang))
        else:
            st.info("Feature importance is not available for Linear Regression (no native importances).")

    # ── Statistical notes ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🔍 Evaluation Methodology & Metric Interpretation</div>',
                unsafe_allow_html=True)
    col_note1, col_note2 = st.columns(2)
    with col_note1:
        best_m = metrics_all.get(best_model, {})
        m_mae = best_m.get("MAE", 8.22)
        m_rmse = best_m.get("RMSE", 126.37)
        m_r2 = best_m.get("R2", 0.9774)
        st.markdown(f"""
        <div class="card">
          <strong style="color:{accent_green};">Understanding MAE vs RMSE Disparity</strong>
          <ul style="margin-top:8px;font-size:0.85rem;color:{text_main};">
            <li><strong>MAE ≈ {m_mae:.2f} t/ha:</strong> Average absolute error across all test predictions (Median AE is only ~0.17 t/ha).</li>
            <li><strong>RMSE ≈ {m_rmse:.2f} t/ha:</strong> Heavily inflated because the top 1% worst errors contribute ~99% of total squared error, caused by <em>Coconut</em> reporting yield in nuts/ha (values up to 21,105) rather than metric tonnes. Non-coconut crops have RMSE < 10 t/ha!</li>
            <li><strong>R² = {m_r2:.4f}:</strong> Explains over 97% of target variance on unseen held-out test data.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_note2:
        st.markdown(f"""
        <div class="card">
          <strong style="color:{accent_green};">Train / Test Split Design</strong>
          <ul style="margin-top:8px;font-size:0.85rem;color:{text_main};">
            <li><strong>Strategy:</strong> {eval_meta.get('split_method', 'Random 80/20, seed=42')}</li>
            <li><strong>Test set:</strong> {eval_meta.get('test_size', 0):,} records ({eval_meta.get('test_size_fraction', 0.2)*100:.0f}%)</li>
            <li><strong>Limitation:</strong> {eval_meta.get('split_limitation', 'Random split does not guarantee chronological generalization.')}</li>
            <li><strong>Trained at:</strong> {eval_meta.get('trained_at', 'N/A')}</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    st.caption("All metrics above are loaded directly from verified evaluation artifacts produced by train_model.py.")

    # ── Coconut Measurement Standardization Box ─────────────────────────────
    st.markdown("""
    <div class="card" style="border-left: 4px solid #3B82F6; margin-top: 14px;">
      <strong style="color:#3B82F6; font-size: 1.05rem;">🥥 Coconut Measurement Standardization & Conversion Analysis</strong>
      <div style="margin-top: 8px; font-size: 0.88rem; line-height: 1.5;">
        <strong>Statistical Target Audit:</strong> Official Indian statistical repositories (Directorate of Economics & Statistics / Coconut Development Board) measure coconut yield in <strong>nuts per hectare</strong>, reaching figures up to 21,105 nuts/ha (national mean ~8,815 nuts/ha). Food crops are recorded in <strong>tonnes per hectare</strong> (mean 1.5–5.0 t/ha).<br><br>
        <strong>Defensible Scientific Conversion:</strong> Under standard FAO and CDB conversion benchmarks (average whole fresh nut weight = <strong>1.4 kg</strong>), converted yield is given by:<br>
        <code>Yield (t/ha) = Yield (nuts/ha) &times; 1.4 / 1000</code><br><br>
        <strong>Architectural Decision:</strong> To maintain strict data integrity without destructive overwriting, CropZen predicts in the official native unit (nuts/ha) and computes the physical mass equivalence (t/ha) at inference time. Non-coconut crops retain strictly standard metric tonnes per hectare.
      </div>
    </div>
    """, unsafe_allow_html=True)

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
          <div style="font-size:0.8rem;color:{text_muted};margin-bottom:6px;">{T('train_last_run', lang)}: {trained_at}</div>
          <div style="font-size:1.2rem;font-weight:700;color:{accent_green};">✅ {T('train_success', lang)}</div>
          <div style="font-size:1rem;color:{text_main};margin-top:4px;">Best Model: <strong>{best}</strong></div>
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
    **Step-by-step local retraining options:**

    **1. Train on Curated Model-Ready Dataset (2000–2020) — Recommended:**
    ```powershell
    .venv\\Scripts\\python.exe train_model.py --dataset model_ready
    ```

    **2. Train on Original Baseline Dataset (1997–2020):**
    ```powershell
    .venv\\Scripts\\python.exe train_model.py --dataset original
    ```

    **3. Train with Time-Based Split (Train ≤2015, Test >2015):**
    ```powershell
    .venv\\Scripts\\python.exe train_model.py --dataset model_ready --split time --split-year 2015
    ```

    **4. Verify the output artifacts exist:**
    ```powershell
    ls models/best_model.pkl
    ls outputs/evaluation.json
    ls outputs/feature_schema.json
    ```

    **5. Restart the Streamlit app to load the new model:**
    ```powershell
    .venv\\Scripts\\python.exe -m streamlit run app.py
    ```

    *Security note: CSV files uploaded through the Batch Prediction interface are used strictly for inference and cannot alter or replace training datasets.*
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
                template=plotly_template,
            )
            fig.update_layout(xaxis_tickangle=-30,
                              font=dict(color=text_main),
                              plot_bgcolor=plotly_bg, paper_bgcolor=plotly_bg)
            st.plotly_chart(style_plotly_chart(fig), use_container_width=True)

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
        st.markdown("### 📂 Data Source Registry & Model Card")
        st.markdown("""
        CropZen integrates **four distinct agricultural datasets**, maintained under a strict data-governance policy to avoid mixing incompatible geographic aggregation levels or introducing data leakage:
        """)

        st.markdown("""
        | Dataset Name | File | Rows | Geography | Years | Role / Scope |
        |---|---|---|---|---|---|
        | **Model-Ready** | `cropzen_model_ready_2000_2020.csv` | 17,983 | State | 2000–2020 | **Model Training & Evaluation** (Area, Rainfall, Fertilizer, Pesticide) |
        | **Original Baseline** | `crop_yield.csv` | 19,689 | State | 1997–2020 | **Historical Benchmark** (contains 1997–1999 baseline records) |
        | **District-Level** | `cropzen_india_district_2000_2020.csv` | 311,488 | District (706) | 2000–2020 | **Geospatial EDA Only** (Missing rainfall, fertilizer, pesticide) |
        | **Official All-India** | `cropzen_india_official_all_india_2021_2025.csv` | 225 | National Aggregate | 2020-21 to 2024-25 | **Official Reporting Only** (Observed Yield missing — no supervised validation) |
        | **Combined Asset** | `cropzen_india_combined_2000_2025.csv` | 311,713 | Mixed | 2000–2025 | **Cross-Source Registry** (Requires grouping by geography_level) |
        """)

        st.markdown("""
        **Validated Model Feature Schema:**
        - **Input Features (8):** `Crop`, `Season`, `State`, `Crop_Year`, `Area` (ha), `Annual_Rainfall` (mm), `Fertilizer` (kg), `Pesticide` (kg)
        - **Target:** `Yield` (log1p-transformed during training; expm1 inverse applied to predictions; metric tonnes/ha)

        **Target Leakage Exclusion:** `Production` is strictly excluded because it represents post-harvest output
        and in many records correlates directly with the target computation (`Yield = Production / Area`),
        causing artificial target leakage.

        **Evaluation Design & Verification:**
        - Trained with scikit-learn Pipelines combining `OrdinalEncoder` (handling unseen categories safely via `-1`) and `StandardScaler`.
        - Held-out test evaluation using MAE, RMSE, and R² (never percentage accuracy).
        - Official 2020-21 to 2024-25 data records do NOT contain observed yield values; no fabricated labels are generated.
        """)

    with tab3:
        st.markdown(f"### {T('about_algorithms', lang)}")

        col_lr, col_dt, col_rf = st.columns(3)
        with col_lr:
            st.markdown(f"""
            <div class="card">
              <strong style="color:{accent_green};">Linear Regression</strong>
              <p style="font-size:0.85rem;color:{text_main};margin-top:8px;">
                Fits a linear relationship between input features and yield.
                Simple, fast, and interpretable. Struggles with non-linear patterns.
                Acts as a baseline for comparison.
              </p>
            </div>
            """, unsafe_allow_html=True)
        with col_dt:
            st.markdown(f"""
            <div class="card">
              <strong style="color:{accent_green};">Decision Tree Regressor</strong>
              <p style="font-size:0.85rem;color:{text_main};margin-top:8px;">
                Learns recursive splits on feature values. Captures non-linear patterns.
                Prone to overfitting without depth limits.
                Provides native feature importance scores.
              </p>
            </div>
            """, unsafe_allow_html=True)
        with col_rf:
            st.markdown(f"""
            <div class="card">
              <strong style="color:{accent_green};">Random Forest Regressor</strong>
              <p style="font-size:0.85rem;color:{text_main};margin-top:8px;">
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
