import html
import os
import joblib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 2. DESIGN SYSTEM & CUSTOM STYLING
# ==========================================================

STATUS_STYLES = {
    "Poor": {
        "bg": "#FEF2F2",
        "border": "#FECACA",
        "accent": "#DC2626",
        "text": "#991B1B",
        "badge_bg": "#FEE2E2",
        "tier": "Poor",
    },
    "Average": {
        "bg": "#FFF7ED",
        "border": "#FED7AA",
        "accent": "#EA580C",
        "text": "#9A3412",
        "badge_bg": "#FFEDD5",
        "tier": "Average",
    },
    "Good": {
        "bg": "#FEFCE8",
        "border": "#FDE047",
        "accent": "#CA8A04",
        "text": "#854D0E",
        "badge_bg": "#FEF9C3",
        "tier": "Good",
    },
    "Excellent": {
        "bg": "#ECFDF5",
        "border": "#6EE7B7",
        "accent": "#10B981",
        "text": "#14532D",
        "badge_bg": "#D1FAE5",
        "tier": "Excellent",
    },
}


def get_status_style(status_text):
    for key, style in STATUS_STYLES.items():
        if key in status_text:
            return style
    return STATUS_STYLES["Excellent"]


def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

        :root {
            --primary: #14532D;
            --secondary: #10B981;
            --accent: #84CC16;
            --bg: #F5F8F5;
            --card: #FFFFFF;
            --text: #1F2937;
            --muted: #6B7280;
            --border: #E5E7EB;
        }

        html, body, [class*="css"], .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #1F2937;
            font-variant-numeric: tabular-nums;
        }

        h1, h2, h3, .hero-title, .agri-page-title, .section-heading {
            font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .stApp {
            background-color: #F5F8F5;
        }

        /* Clean SaaS chrome */
        #MainMenu, footer {
            visibility: hidden;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        /* Main container layout */
        .main .block-container {
            max-width: 1200px;
            padding: 1.75rem 2.25rem 3rem 2.25rem;
        }

        @media (max-width: 768px) {
            .main .block-container {
                padding: 1.15rem 1rem 2rem 1rem;
            }
        }

        /* Distinct Deep Forest Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #14532D 0%, #0F3E22 100%);
            border-right: 1px solid #0B311A;
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.25rem;
            padding-bottom: 1.5rem;
        }

        section[data-testid="stSidebar"] button[kind="header"],
        section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] svg {
            color: #ECFDF5 !important;
            fill: #ECFDF5 !important;
        }

        .sidebar-brand {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 14px;
            padding: 16px;
            margin-bottom: 18px;
        }

        .sidebar-brand-row {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .sidebar-logo {
            width: 42px;
            height: 42px;
            border-radius: 11px;
            background: linear-gradient(135deg, #10B981 0%, #84CC16 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.35);
        }

        .sidebar-brand-title {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 1.02rem;
            font-weight: 700;
            color: #FFFFFF;
            line-height: 1.2;
        }

        .sidebar-brand-sub {
            font-size: 0.76rem;
            color: #BBF7D0;
            line-height: 1.35;
            margin-top: 3px;
        }

        .sidebar-section-label {
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #86EFAC;
            margin: 6px 4px 8px 4px;
        }

        /* Sidebar Nav Items */
        section[data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 6px;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            background: transparent;
            border: 1px solid transparent;
            border-radius: 10px;
            padding: 10px 13px !important;
            margin: 0 !important;
            width: 100%;
            transition: background-color 0.18s ease, border-color 0.18s ease, transform 0.14s ease;
            cursor: pointer;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background-color: rgba(255, 255, 255, 0.09);
            border-color: rgba(255, 255, 255, 0.12);
            transform: translateX(2px);
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: rgba(16, 185, 129, 0.22);
            border-color: rgba(132, 204, 22, 0.45);
            box-shadow: inset 3.5px 0 0 #84CC16;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
            display: none !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label p {
            font-size: 0.93rem !important;
            font-weight: 500 !important;
            color: #DCFCE7 !important;
            margin: 0 !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
            font-weight: 700 !important;
            color: #FFFFFF !important;
        }

        .sidebar-footer-card {
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            padding: 12px 14px;
            margin-top: 14px;
        }

        .sidebar-footer-title {
            font-size: 0.8rem;
            font-weight: 700;
            color: #84CC16;
            margin-bottom: 3px;
        }

        .sidebar-footer-sub {
            font-size: 0.77rem;
            color: #D1FAE5;
            line-height: 1.4;
        }

        /* Page Header */
        .agri-page-header {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-left: 5px solid #14532D;
            border-radius: 16px;
            padding: 20px 24px;
            margin-bottom: 1.4rem;
            box-shadow: 0 1px 3px rgba(17, 24, 39, 0.04), 0 4px 14px rgba(20, 83, 45, 0.03);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
        }

        .agri-page-header-left {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .agri-page-icon {
            width: 50px;
            height: 50px;
            border-radius: 13px;
            background: linear-gradient(135deg, #ECFDF5 0%, #DCFCE7 100%);
            border: 1px solid #A7F3D0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            flex-shrink: 0;
        }

        .agri-page-title {
            font-size: 1.6rem;
            font-weight: 800;
            color: #14532D;
            margin: 0;
            line-height: 1.2;
            letter-spacing: -0.02em;
        }

        .agri-page-desc {
            font-size: 0.94rem;
            color: #6B7280;
            margin: 4px 0 0 0;
            line-height: 1.45;
        }

        .agri-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #ECFDF5;
            color: #14532D;
            border: 1px solid #A7F3D0;
            border-radius: 999px;
            padding: 6px 13px;
            font-size: 0.77rem;
            font-weight: 700;
            white-space: nowrap;
        }

        @media (max-width: 640px) {
            .agri-page-header {
                flex-direction: column;
                align-items: flex-start;
                padding: 16px 18px;
            }
            .agri-page-title {
                font-size: 1.32rem;
            }
        }

        /* Executive Hero Banner */
        .hero-gradient-banner {
            background: linear-gradient(135deg, #14532D 0%, #166534 58%, #0D6E48 100%);
            border-radius: 20px;
            padding: 32px 34px;
            color: #FFFFFF;
            box-shadow: 0 10px 30px rgba(20, 83, 45, 0.16);
            margin-bottom: 1.25rem;
            position: relative;
            overflow: hidden;
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(132, 204, 22, 0.18);
            color: #D9F99D;
            border: 1px solid rgba(132, 204, 22, 0.4);
            border-radius: 999px;
            padding: 5px 13px;
            font-size: 0.77rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin-bottom: 14px;
        }

        .hero-title {
            font-size: 2.3rem;
            font-weight: 800;
            color: #FFFFFF;
            line-height: 1.12;
            margin: 0 0 8px 0;
            letter-spacing: -0.025em;
        }

        .hero-subtitle {
            font-size: 1.08rem;
            font-weight: 600;
            color: #A7F3D0;
            margin: 0 0 14px 0;
        }

        .hero-body {
            font-size: 0.96rem;
            color: #ECFDF5;
            line-height: 1.65;
            margin: 0;
            opacity: 0.94;
        }

        /* Cards */
        .agri-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 20px 22px;
            box-shadow: 0 1px 3px rgba(17, 24, 39, 0.04);
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
            height: 100%;
        }

        .agri-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 22px rgba(20, 83, 45, 0.07);
            border-color: #D1FAE5;
        }

        .kpi-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-top: 3px solid #10B981;
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 1px 3px rgba(17, 24, 39, 0.04);
            display: flex;
            flex-direction: column;
            gap: 6px;
            height: 100%;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }

        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(20, 83, 45, 0.07);
        }

        .kpi-label {
            font-size: 0.82rem;
            font-weight: 600;
            color: #6B7280;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .kpi-value {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 1.7rem;
            font-weight: 800;
            color: #14532D;
            line-height: 1.2;
            word-break: break-word;
        }

        .kpi-sub {
            font-size: 0.78rem;
            color: #6B7280;
        }

        /* Feature & Navigation Cards */
        .feature-item-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            padding: 13px 16px;
            display: flex;
            align-items: center;
            gap: 12px;
            box-shadow: 0 1px 2px rgba(17, 24, 39, 0.03);
            margin-bottom: 10px;
            transition: transform 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease;
        }

        .feature-item-card:hover {
            transform: translateY(-1px);
            border-color: #10B981;
            box-shadow: 0 6px 14px rgba(20, 83, 45, 0.06);
        }

        .feature-item-icon {
            width: 38px;
            height: 38px;
            border-radius: 10px;
            background: #ECFDF5;
            border: 1px solid #A7F3D0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.05rem;
            flex-shrink: 0;
        }

        .feature-item-text {
            font-size: 0.93rem;
            font-weight: 600;
            color: #1F2937;
            line-height: 1.35;
        }

        /* Section Headings */
        .section-heading {
            font-size: 1.15rem;
            font-weight: 700;
            color: #14532D;
            margin: 0.35rem 0 0.75rem 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .section-subtext {
            font-size: 0.92rem;
            color: #6B7280;
            margin: -0.35rem 0 0.85rem 0;
            line-height: 1.5;
        }

        .input-group-label {
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #10B981;
            margin-bottom: 8px;
        }

        /* Prediction Highlight Spotlight */
        .prediction-hero-card {
            background: linear-gradient(135deg, #14532D 0%, #166534 70%, #0F766E 100%);
            border-radius: 18px;
            padding: 24px 26px;
            color: #FFFFFF;
            box-shadow: 0 10px 25px rgba(20, 83, 45, 0.16);
            height: 100%;
        }

        .prediction-eyebrow {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #A7F3D0;
            margin-bottom: 8px;
        }

        .prediction-number {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 2.55rem;
            font-weight: 800;
            color: #FFFFFF;
            line-height: 1.08;
            margin-bottom: 8px;
        }

        .prediction-unit {
            font-size: 1.05rem;
            font-weight: 600;
            color: #84CC16;
        }

        .prediction-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 14px;
        }

        .prediction-pill {
            background: rgba(255, 255, 255, 0.14);
            border: 1px solid rgba(255, 255, 255, 0.22);
            border-radius: 999px;
            padding: 4px 12px;
            font-size: 0.8rem;
            font-weight: 600;
            color: #FFFFFF;
        }

        .status-result-card {
            border-radius: 18px;
            padding: 22px 24px;
            box-shadow: 0 4px 14px rgba(17, 24, 39, 0.04);
            height: 100%;
        }

        .status-pill-badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 999px;
            font-size: 0.94rem;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .tier-track {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 6px;
            margin: 10px 0 12px 0;
        }

        .tier-step {
            text-align: center;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 5px 4px;
            border-radius: 7px;
            background: rgba(255, 255, 255, 0.65);
            border: 1px solid rgba(0, 0, 0, 0.08);
            color: #6B7280;
        }

        .tier-step-active {
            font-weight: 800;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
        }

        .status-message-text {
            font-size: 0.93rem;
            line-height: 1.5;
            margin: 0 0 12px 0;
            font-weight: 500;
        }

        .status-meta-row {
            font-size: 0.83rem;
            padding-top: 10px;
            border-top: 1px solid rgba(0, 0, 0, 0.08);
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 8px;
        }

        /* Model Comparison Cards */
        .model-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: 0 1px 3px rgba(17, 24, 39, 0.04);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
            height: 100%;
        }

        .model-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(20, 83, 45, 0.08);
        }

        .model-card-best {
            background: linear-gradient(180deg, #ECFDF5 0%, #FFFFFF 45%);
            border: 1.5px solid #10B981;
            box-shadow: 0 6px 18px rgba(16, 185, 129, 0.12);
        }

        .model-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 12px;
        }

        .model-card-name {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 0.98rem;
            font-weight: 700;
            color: #14532D;
        }

        .model-best-pill {
            background: #14532D;
            color: #D9F99D;
            font-size: 0.7rem;
            font-weight: 700;
            padding: 3px 9px;
            border-radius: 999px;
        }

        .model-metric-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            padding: 5px 0;
            border-bottom: 1px solid #F3F4F6;
        }

        .model-metric-row:last-child {
            border-bottom: none;
        }

        /* Insight Cards */
        .insight-list-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 1px 3px rgba(17, 24, 39, 0.04);
            height: 100%;
        }

        .insight-list-title {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 1.02rem;
            font-weight: 700;
            color: #14532D;
            margin: 0 0 12px 0;
        }

        .insight-item {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            padding: 10px 12px;
            border-radius: 10px;
            background: #F5F8F5;
            border: 1px solid #E5E7EB;
            margin-bottom: 8px;
            font-size: 0.91rem;
            color: #1F2937;
            line-height: 1.48;
        }

        .insight-item:last-child {
            margin-bottom: 0;
        }

        /* Callout Banner */
        .agri-callout {
            background: #ECFDF5;
            border: 1px solid #A7F3D0;
            border-left: 4px solid #10B981;
            border-radius: 12px;
            padding: 14px 18px;
            font-size: 0.92rem;
            color: #1F2937;
            line-height: 1.55;
            margin: 0.75rem 0;
        }

        /* Empty State */
        .empty-state-card {
            background: #FFFFFF;
            border: 1.5px dashed #10B981;
            border-radius: 18px;
            padding: 38px 28px;
            text-align: center;
            max-width: 680px;
            margin: 1.5rem auto;
            box-shadow: 0 4px 14px rgba(20, 83, 45, 0.04);
        }

        .empty-state-icon {
            font-size: 2.6rem;
            margin-bottom: 10px;
        }

        .empty-state-title {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
            color: #14532D;
            margin-bottom: 8px;
        }

        .empty-state-text {
            font-size: 0.95rem;
            color: #6B7280;
            line-height: 1.6;
        }

        /* Native Streamlit Component Polish */
        /* Support both old and new Streamlit border-wrapper selectors */
        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stVerticalBlock"] > div[style*="border"] {
            background-color: #FFFFFF;
            border-color: #E5E7EB !important;
            border-radius: 16px !important;
            box-shadow: 0 1px 3px rgba(17, 24, 39, 0.04);
        }

        div[data-testid="stMetric"] {
            background-color: #F9FBF9;
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            padding: 12px 14px;
        }

        div[data-testid="stMetric"] label {
            color: #6B7280 !important;
            font-weight: 600 !important;
            font-size: 0.82rem !important;
        }

        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #14532D !important;
            font-weight: 800 !important;
        }

        .stButton > button {
            border-radius: 11px !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.2rem !important;
            transition: all 0.18s ease !important;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #14532D 0%, #10B981 100%) !important;
            border: none !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 12px rgba(20, 83, 45, 0.22) !important;
        }

        .stButton > button[kind="primary"]:hover {
            filter: brightness(1.06);
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(20, 83, 45, 0.28) !important;
        }

        .stButton > button[kind="secondary"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E5E7EB !important;
            color: #14532D !important;
        }

        .stButton > button[kind="secondary"]:hover {
            background-color: #ECFDF5 !important;
            border-color: #10B981 !important;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            border-radius: 10px !important;
            border-color: #D1D5DB !important;
            background-color: #FFFFFF !important;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }

        div[data-baseweb="select"] > div:focus-within,
        div[data-baseweb="input"] > div:focus-within {
            border-color: #10B981 !important;
            box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.16) !important;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            overflow: hidden;
        }

        /* Dataframe header styling */
        div[data-testid="stDataFrame"] th {
            background-color: #F5F8F5 !important;
            color: #14532D !important;
            font-weight: 700 !important;
            font-size: 0.82rem !important;
        }

        /* Section heading inside bordered container gets breathing room */
        div[data-testid="stVerticalBlockBorderWrapper"] .section-heading,
        div[data-testid="stVerticalBlock"] .section-heading {
            margin-top: 0.1rem;
        }

        /* Spinner text */
        div[data-testid="stSpinner"] p {
            color: #14532D !important;
            font-weight: 600 !important;
        }

        /* Altair chart container */
        div[data-testid="stArrowVegaLiteChart"] {
            border-radius: 10px;
            overflow: hidden;
        }

        /* Success/warning/info boxes */
        div[data-testid="stAlert"] {
            border-radius: 10px !important;
        }

        /* Expander */
        details summary {
            font-weight: 600 !important;
            color: #14532D !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(icon, title, description, badge=None):
    badge_html = (
        f'<span class="agri-badge">{html.escape(badge)}</span>'
        if badge
        else ""
    )
    st.markdown(
        f"""
        <div class="agri-page-header">
            <div class="agri-page-header-left">
                <div class="agri-page-icon">{icon}</div>
                <div>
                    <h1 class="agri-page-title">{html.escape(title)}</h1>
                    <p class="agri-page-desc">{html.escape(description)}</p>
                </div>
            </div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label, value, subtext="", icon="", accent_color="#10B981"):
    icon_prefix = f"{icon} " if icon else ""
    sub_html = (
        f'<div class="kpi-sub">{html.escape(str(subtext))}</div>'
        if subtext
        else ""
    )
    st.markdown(
        f"""
        <div class="kpi-card" style="border-top-color: {accent_color};">
            <div class="kpi-label">{icon_prefix}{html.escape(str(label))}</div>
            <div class="kpi-value">{html.escape(str(value))}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_card(title, description, icon=""):
    icon_prefix = f"{icon} " if icon else ""
    st.markdown(
        f"""
        <div class="agri-card">
            <div class="insight-list-title" style="margin-bottom: 6px;">{icon_prefix}{html.escape(str(title))}</div>
            <div style="font-size: 0.92rem; color: #4B5563; line-height: 1.55;">{html.escape(str(description))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_list_card(title, items, bullet_icon="•"):
    items_html = "".join(
        f'<div class="insight-item"><span>{bullet_icon}</span><span>{html.escape(str(item))}</span></div>'
        for item in items
    )
    st.markdown(
        f"""
        <div class="insight-list-card">
            <div class="insight-list-title">{html.escape(title)}</div>
            {items_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def navigate_to(target_page):
    st.session_state["nav_page"] = target_page


# ==========================================================
# 3. PROJECT PATHS
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "crop_yield_model.pkl"
)

DATA_DIR = os.path.join(
    BASE_DIR, "data", "raw"
)


# ==========================================================
# 4. LOAD DATASET
# ==========================================================

if not os.path.exists(DATA_DIR):
    st.error("Data folder not found!")
    st.code(DATA_DIR)
    st.stop()

csv_files = sorted([
    file for file in os.listdir(DATA_DIR)
    if file.lower().endswith(".csv")
])

if not csv_files:
    st.error("No CSV file found inside data/raw!")
    st.stop()

data_filename = (
    "yield_df.csv"
    if "yield_df.csv" in csv_files
    else csv_files[0]
)

DATA_PATH = os.path.join(DATA_DIR, data_filename)

try:
    df = pd.read_csv(DATA_PATH)
    df.drop(
        columns=["Unnamed: 0"],
        errors="ignore",
        inplace=True
    )

except Exception as error:
    st.error("Unable to load dataset.")
    st.exception(error)
    st.stop()


# ==========================================================
# 5. VALIDATE DATASET
# ==========================================================

required_columns = [
    "Area",
    "Item",
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
    "hg/ha_yield"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error("Required columns are missing!")
    st.write(missing_columns)
    st.stop()


# ==========================================================
# 6. DATA CLEANING
# ==========================================================

numeric_columns = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
    "hg/ha_yield"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df.dropna(
    subset=required_columns,
    inplace=True
)

df["Area"] = df["Area"].astype(str)
df["Item"] = df["Item"].astype(str)

df = df[df["hg/ha_yield"] >= 0].copy()

if df.empty:
    st.error("No valid records found in the dataset.")
    st.stop()


# ==========================================================
# 7. LOAD TRAINED MODEL
# ==========================================================

if not os.path.exists(MODEL_PATH):
    st.error("Trained model file not found!")
    st.code(MODEL_PATH)
    st.stop()

try:
    model = joblib.load(MODEL_PATH)

except Exception as error:
    st.error("Unable to load trained model.")
    st.exception(error)
    st.stop()


# ==========================================================
# 8. CREATE DROPDOWN OPTIONS
# ==========================================================

crop_options = sorted(
    df["Item"].unique().tolist()
)

area_options = sorted(
    df["Area"].unique().tolist()
)


# ==========================================================
# 9. CROP-SPECIFIC YIELD THRESHOLDS
# ==========================================================

def get_crop_thresholds(crop_name):

    crop_data = df.loc[
        df["Item"] == crop_name,
        "hg/ha_yield"
    ].dropna()

    if crop_data.empty:
        crop_data = df["hg/ha_yield"]

    return {
        "q25": float(crop_data.quantile(0.25)),
        "q50": float(crop_data.quantile(0.50)),
        "q75": float(crop_data.quantile(0.75)),
        "average": float(crop_data.mean()),
        "count": len(crop_data)
    }


def classify_yield(value, thresholds):

    if value < thresholds["q25"]:
        return (
            "🔴 Poor",
            "The predicted yield is in the lower "
            "quarter of this crop's historical results."
        )

    elif value < thresholds["q50"]:
        return (
            "🟠 Average",
            "The predicted yield is below the "
            "historical middle value for this crop."
        )

    elif value < thresholds["q75"]:
        return (
            "🟡 Good",
            "The predicted yield is above the "
            "historical middle value for this crop."
        )

    else:
        return (
            "🟢 Excellent",
            "The predicted yield is in the upper "
            "quarter of this crop's historical results."
        )


# ==========================================================
# 10. SELECT HISTORICAL COMPARISON DATA
# ==========================================================

def get_reference_data(crop_name, area_name):

    crop_area_data = df[
        (df["Item"] == crop_name) &
        (df["Area"] == area_name)
    ].copy()

    crop_data = df[
        df["Item"] == crop_name
    ].copy()

    # Prefer the selected crop and area.

    if len(crop_area_data) >= 8:
        return crop_area_data, "Selected crop and area"

    # If insufficient, compare all areas for the crop.

    if len(crop_data) >= 8:
        return crop_data, "Selected crop across all areas"

    # Last fallback.

    return df.copy(), "Entire dataset (limited crop history)"


# ==========================================================
# 11. GENERATE CROP INSIGHTS
# ==========================================================

def generate_crop_insights(
    crop_name,
    area_name,
    rainfall,
    temperature,
    pesticides,
    predicted_yield
):

    reference, reference_label = get_reference_data(
        crop_name,
        area_name
    )

    high_yield_cutoff = reference[
        "hg/ha_yield"
    ].quantile(0.75)

    high_yield_records = reference[
        reference["hg/ha_yield"] >= high_yield_cutoff
    ]

    if high_yield_records.empty:
        high_yield_records = reference

    input_values = {
        "Rainfall": (
            rainfall,
            "average_rain_fall_mm_per_year",
            "mm/year"
        ),

        "Temperature": (
            temperature,
            "avg_temp",
            "°C"
        ),

        "Pesticide usage": (
            pesticides,
            "pesticides_tonnes",
            "tonnes"
        )
    }

    possible_factors = []
    recommendations = []
    comparison_rows = []

    for name, (user_value, column, unit) in input_values.items():

        historical = pd.to_numeric(
            reference[column],
            errors="coerce"
        ).dropna()

        high_group = pd.to_numeric(
            high_yield_records[column],
            errors="coerce"
        ).dropna()

        if historical.empty:
            continue

        q25 = historical.quantile(0.25)
        median = historical.median()
        q75 = historical.quantile(0.75)

        high_median = (
            high_group.median()
            if not high_group.empty
            else median
        )

        if user_value < q25:

            comparison = "Below usual range"

            possible_factors.append(
                f"{name} is lower than the usual "
                "historical range for this comparison group."
            )

            if name == "Rainfall":

                recommendations.append(
                    "Check whether the crop is receiving "
                    "enough water. Review irrigation "
                    "availability and the crop's water needs."
                )

            elif name == "Temperature":

                recommendations.append(
                    "Check whether the temperature is "
                    "suitable for this crop's growing period."
                )

            else:

                recommendations.append(
                    "Monitor pests and diseases. Check "
                    "whether pest control is needed before "
                    "making pesticide-use decisions."
                )

        elif user_value > q75:

            comparison = "Above usual range"

            possible_factors.append(
                f"{name} is higher than the usual "
                "historical range for this comparison group."
            )

            if name == "Rainfall":

                recommendations.append(
                    "Check field drainage and watch for "
                    "excess water or waterlogging."
                )

            elif name == "Temperature":

                recommendations.append(
                    "Check for signs of heat stress and "
                    "whether the crop is suitable for "
                    "these temperature conditions."
                )

            else:

                recommendations.append(
                    "Check whether the pesticide and its "
                    "quantity are suitable for this crop. "
                    "Avoid unnecessary pesticide use."
                )

        else:

            comparison = "Within usual range"

        comparison_rows.append({

            "Parameter": name,

            "Your Input": (
                f"{user_value:,.2f} {unit}"
            ),

            "Historical Lower Level": (
                f"{q25:,.2f} {unit}"
            ),

            "Historical Middle Value": (
                f"{median:,.2f} {unit}"
            ),

            "Historical Upper Level": (
                f"{q75:,.2f} {unit}"
            ),

            "Result": comparison

        })

    if not possible_factors:

        possible_factors.append(
            "Rainfall, temperature and pesticide usage "
            "are within their usual historical ranges. "
            "Other farming conditions may still affect yield."
        )

    if predicted_yield >= high_yield_cutoff:

        yield_context = (
            "The predicted yield is at or above the "
            "upper quarter of historical yields in "
            "the comparison group."
        )

        recommendations.append(
            "Continue monitoring crop health and field "
            "conditions. Compare the actual harvest "
            "with this prediction."
        )

    else:

        yield_context = (
            "The predicted yield is below the upper "
            "quarter of historical yields in the "
            "comparison group."
        )

        recommendations.append(
            "Check soil quality, seed variety, irrigation, "
            "sowing time, pest problems and farming methods."
        )

    recommendations = list(
        dict.fromkeys(recommendations)
    )

    return {

        "reference_label": reference_label,

        "reference_count": len(reference),

        "high_yield_cutoff": float(high_yield_cutoff),

        "yield_context": yield_context,

        "possible_factors": possible_factors,

        "recommendations": recommendations,

        "comparison_table": pd.DataFrame(comparison_rows)

    }


# ==========================================================
# 12. INJECT STYLING & SIDEBAR NAVIGATION
# ==========================================================

inject_custom_css()

NAV_PAGES = [
    "🏠 Home",
    "🔮 Predict Yield",
    "🔀 What-if Analysis",
    "📊 Data Analysis",
    "🤖 Model Performance",
    "ℹ️ About",
    "💰 Profit Calculator",
]

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = NAV_PAGES[0]

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-row">
                <div class="sidebar-logo">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 20V9" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round"/>
                        <path d="M12 13C12 8.5 15.5 5 20 5C20 9.5 16.5 13 12 13Z" fill="#FFFFFF"/>
                        <path d="M12 16C12 11.5 8.5 8 4 8C4 12.5 7.5 16 12 16Z" fill="#ECFDF5"/>
                    </svg>
                </div>
                <div>
                    <div class="sidebar-brand-title">Crop Yield Prediction</div>
                    <div class="sidebar-brand-sub">Machine Learning Based Agricultural Analysis</div>
                </div>
            </div>
        </div>
        <div class="sidebar-section-label">Workspace Navigation</div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        NAV_PAGES,
        key="nav_page",
        label_visibility="collapsed",
    )

    st.markdown(
        f"""
        <div class="sidebar-footer-card">
            <div class="sidebar-footer-title">Yield unit: hg/ha</div>
            <div class="sidebar-footer-sub">
                {df["Item"].nunique()} crops • {df["Area"].nunique()} areas • {len(df):,} records
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# 13. HOME PAGE
# ==========================================================

if page == "🏠 Home":

    hero_svg = """
    <svg viewBox="0 0 380 235" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Agricultural AI dashboard graphic">
        <rect x="6" y="6" width="368" height="223" rx="18" fill="rgba(255,255,255,0.08)" stroke="rgba(255,255,255,0.18)" stroke-width="1.5"/>
        <circle cx="312" cy="54" r="24" fill="rgba(132,204,22,0.22)" stroke="rgba(132,204,22,0.5)" stroke-width="1.5"/>
        <circle cx="312" cy="54" r="12" fill="#84CC16" opacity="0.75"/>
        <path d="M8 175 Q 110 132 220 166 T 372 152 L 372 227 L 8 227 Z" fill="rgba(16,185,129,0.28)"/>
        <path d="M8 194 Q 140 160 250 188 T 372 180 L 372 227 L 8 227 Z" fill="rgba(132,204,22,0.26)"/>
        <g stroke="#ECFDF5" stroke-width="2.2" stroke-linecap="round">
            <line x1="66" y1="198" x2="66" y2="128"/>
            <line x1="106" y1="194" x2="106" y2="112"/>
            <line x1="146" y1="198" x2="146" y2="132"/>
        </g>
        <path d="M66 158 C48 152, 44 138, 60 138 C65 138, 66 148, 66 158 Z" fill="#84CC16"/>
        <path d="M66 146 C84 140, 88 126, 72 126 C67 126, 66 136, 66 146 Z" fill="#10B981"/>
        <path d="M106 148 C86 142, 82 126, 100 126 C105 126, 106 138, 106 148 Z" fill="#10B981"/>
        <path d="M106 134 C126 128, 130 112, 112 112 C107 112, 106 124, 106 134 Z" fill="#84CC16"/>
        <path d="M146 162 C128 156, 124 142, 140 142 C145 142, 146 152, 146 162 Z" fill="#84CC16"/>
        <path d="M146 150 C164 144, 168 130, 152 130 C147 130, 146 140, 146 150 Z" fill="#10B981"/>
        <rect x="192" y="38" width="150" height="94" rx="12" fill="#FFFFFF" stroke="#A7F3D0" stroke-width="1.5"/>
        <text x="206" y="59" font-family="sans-serif" font-size="10.5" font-weight="700" fill="#14532D">Yield Forecast (hg/ha)</text>
        <rect x="206" y="100" width="15" height="18" rx="3" fill="#D1FAE5"/>
        <rect x="227" y="90" width="15" height="28" rx="3" fill="#6EE7B7"/>
        <rect x="248" y="80" width="15" height="38" rx="3" fill="#10B981"/>
        <rect x="269" y="70" width="15" height="48" rx="3" fill="#14532D"/>
        <path d="M212 94 L235 84 L256 75 L277 64 L326 57" fill="none" stroke="#84CC16" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="326" cy="57" r="4.5" fill="#14532D" stroke="#84CC16" stroke-width="2"/>
    </svg>
    """

    st.markdown(
        f"""
        <div class="hero-gradient-banner">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 28px; align-items: center;">
                <div>
                    <div class="hero-kicker">🌾 Smart Agriculture • AI Analytics Platform</div>
                    <h1 class="hero-title">Crop Yield Prediction</h1>
                    <div class="hero-subtitle">Machine Learning Based Crop Yield Prediction System</div>
                    <p class="hero-body">
                        This application uses Machine Learning to predict crop yield based on agricultural and environmental conditions.
                    </p>
                    <p class="hero-body" style="margin-top: 8px;">
                        It also compares user inputs with historical records to identify possible factors worth investigating.
                    </p>
                </div>
                <div>{hero_svg}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    with nav_col1:
        st.button(
            "🔮 Predict Yield",
            type="primary",
            use_container_width=True,
            on_click=navigate_to,
            args=("🔮 Predict Yield",),
        )
    with nav_col2:
        st.button(
            "📊 Data Analysis",
            type="secondary",
            use_container_width=True,
            on_click=navigate_to,
            args=("📊 Data Analysis",),
        )
    with nav_col3:
        st.button(
            "🤖 Model Performance",
            type="secondary",
            use_container_width=True,
            on_click=navigate_to,
            args=("🤖 Model Performance",),
        )
    with nav_col4:
        st.button(
            "ℹ️ About",
            type="secondary",
            use_container_width=True,
            on_click=navigate_to,
            args=("ℹ️ About",),
        )

    st.write("")

    col1, col2, col3 = st.columns(3)
    with col1:
        render_kpi_card("Crops", df["Item"].nunique(), "Supported crop varieties", "🌱", "#10B981")
    with col2:
        render_kpi_card("Areas", df["Area"].nunique(), "Countries & geographical regions", "🌍", "#14532D")
    with col3:
        render_kpi_card("Records", f"{len(df):,}", "Historical agricultural observations", "📊", "#84CC16")

    st.write("")
    st.markdown('<div class="section-heading">📌 Project Features</div>', unsafe_allow_html=True)

    features = [
        ("🌾", "Crop yield prediction"),
        ("🌍", "Crop and area selection"),
        ("🏷️", "Crop-specific yield classification"),
        ("📈", "Historical yield comparison"),
        ("🌧️", "Rainfall analysis"),
        ("🌡️", "Temperature analysis"),
        ("🤖", "Model performance evaluation"),
        ("📊", "Prediction-based data analysis"),
        ("🔍", "Possible reasons for low or high yield"),
        ("💡", "Suggestions for further investigation"),
    ]

    feat_col1, feat_col2 = st.columns(2)
    for idx, (feat_icon, feat_label) in enumerate(features):
        target_col = feat_col1 if idx % 2 == 0 else feat_col2
        with target_col:
            st.markdown(
                f"""
                <div class="feature-item-card">
                    <div class="feature-item-icon">{feat_icon}</div>
                    <div class="feature-item-text">{html.escape(feat_label)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="agri-callout">
            ℹ️ <strong>Note:</strong> Insights are based on historical patterns. They do not prove cause and effect.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# 14. PREDICT YIELD PAGE
# ==========================================================

elif page == "🔮 Predict Yield":

    render_page_header(
        "🔮",
        "Predict Crop Yield",
        "Enter agricultural information to predict crop yield.",
        badge="ML Inference • hg/ha",
    )

    with st.container(border=True):
        st.markdown('<div class="section-heading">🌱 Crop Information</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-group-label">Crop, Location &amp; Timeline</div>', unsafe_allow_html=True)
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        with row1_col1:
            crop = st.selectbox(
                "Select Crop",
                crop_options
            )
        with row1_col2:
            area = st.selectbox(
                "Select Area / Country",
                area_options
            )
        with row1_col3:
            year = st.number_input(
                "Year",
                min_value=1961,
                max_value=2030,
                value=2025,
                step=1
            )

        st.markdown('<div class="input-group-label" style="margin-top: 10px;">Climate &amp; Field Conditions</div>', unsafe_allow_html=True)
        row2_col1, row2_col2, row2_col3 = st.columns(3)
        with row2_col1:
            rainfall = st.number_input(
                "Rainfall (mm/year)",
                min_value=0.0,
                value=1200.0,
                step=10.0
            )
        with row2_col2:
            temperature = st.number_input(
                "Temperature (°C)",
                min_value=-10.0,
                max_value=60.0,
                value=28.0,
                step=0.5
            )
        with row2_col3:
            pesticides = st.number_input(
                "Pesticides (tonnes)",
                min_value=0.0,
                value=1000.0,
                step=100.0
            )

        st.write("")
        predict_clicked = st.button(
            "🌾 Predict Yield",
            type="primary",
            use_container_width=True,
        )

    if predict_clicked:

        input_data = pd.DataFrame({

            "Area": [area],

            "Item": [crop],

            "Year": [year],

            "average_rain_fall_mm_per_year": [
                rainfall
            ],

            "pesticides_tonnes": [
                pesticides
            ],

            "avg_temp": [
                temperature
            ]

        })

        try:

            # MODEL PREDICTION

            prediction = model.predict(input_data)

            predicted_yield = float(prediction[0])

            if not np.isfinite(predicted_yield):
                raise ValueError(
                    "The model returned an invalid prediction."
                )

            # CROP-SPECIFIC CLASSIFICATION

            thresholds = get_crop_thresholds(crop)

            status, status_message = classify_yield(
                predicted_yield,
                thresholds
            )

            # GENERATE INSIGHTS

            insights = generate_crop_insights(
                crop,
                area,
                rainfall,
                temperature,
                pesticides,
                predicted_yield
            )

            # SAVE LATEST PREDICTION FOR DATA ANALYSIS PAGE

            st.session_state["last_prediction"] = {

                "crop": crop,

                "area": area,

                "year": int(year),

                "rainfall": rainfall,

                "temperature": temperature,

                "pesticides": pesticides,

                "predicted_yield": predicted_yield,

                "status": status,

                "status_message": status_message,

                "thresholds": thresholds,

                "insights": insights

            }

            st.success(
                "✅ Prediction completed successfully!"
            )

            # --------------------------------------------------
            # PREDICTED YIELD & YIELD STATUS CARDS
            # --------------------------------------------------

            status_style = get_status_style(status)
            res_col1, res_col2 = st.columns([1.15, 1])

            with res_col1:
                st.markdown(
                    f"""
                    <div class="prediction-hero-card">
                        <div class="prediction-eyebrow">🌾 Predicted Crop Yield</div>
                        <div class="prediction-number">
                            {predicted_yield:,.2f} <span class="prediction-unit">hg/ha</span>
                        </div>
                        <div class="prediction-pills">
                            <span class="prediction-pill">🌱 {html.escape(crop)}</span>
                            <span class="prediction-pill">🌍 {html.escape(area)}</span>
                            <span class="prediction-pill">📅 {int(year)}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with res_col2:
                tiers_html = ""
                for tier_name in ["Poor", "Average", "Good", "Excellent"]:
                    if tier_name == status_style["tier"]:
                        tiers_html += (
                            f'<div class="tier-step tier-step-active" '
                            f'style="background: {status_style["badge_bg"]}; '
                            f'border-color: {status_style["accent"]}; '
                            f'color: {status_style["text"]};">{tier_name}</div>'
                        )
                    else:
                        tiers_html += f'<div class="tier-step">{tier_name}</div>'

                st.markdown(
                    f"""
                    <div class="status-result-card" style="background: {status_style['bg']}; border: 1.5px solid {status_style['border']}; color: {status_style['text']};">
                        <div class="prediction-eyebrow" style="color: {status_style['text']}; opacity: 0.85;">📊 Yield Status</div>
                        <div class="status-pill-badge" style="background: {status_style['badge_bg']}; border: 1px solid {status_style['border']}; color: {status_style['text']};">
                            {html.escape(status)}
                        </div>
                        <div class="tier-track">{tiers_html}</div>
                        <p class="status-message-text">{html.escape(status_message)}</p>
                        <div class="status-meta-row">
                            <span><strong>{html.escape(crop)} Historical Average:</strong> {thresholds['average']:,.2f} hg/ha</span>
                            <span><strong>Records:</strong> {thresholds['count']:,}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(
                f"Classification uses {thresholds['count']:,} "
                f"historical records for {crop}."
            )

            # --------------------------------------------------
            # PREDICTION INTERPRETATION
            # --------------------------------------------------

            with st.container(border=True):
                st.markdown('<div class="section-heading">📌 Prediction Interpretation</div>', unsafe_allow_html=True)
                st.write(
                    f"""
                    The model predicts **{predicted_yield:,.2f} hg/ha**
                    for **{crop}** in **{area}** during **{int(year)}**.

                    The predicted result is classified as **{status}**
                    based on the historical yield distribution of
                    the selected crop.
                    """
                )

            # --------------------------------------------------
            # CROP YIELD INSIGHTS & RECOMMENDATIONS
            # --------------------------------------------------

            st.markdown('<div class="section-heading">🔍 Crop Yield Insights &amp; Recommendations</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-subtext">This section compares your input values with historical records to identify conditions that may deserve attention.</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"Comparison basis: {insights['reference_label']} "
                f"({insights['reference_count']:,} records)."
            )

            ins_col1, ins_col2 = st.columns(2)
            with ins_col1:
                render_insight_list_card(
                    "Possible Factors to Investigate",
                    insights["possible_factors"],
                    bullet_icon="🔎",
                )
            with ins_col2:
                render_insight_list_card(
                    "💡 Recommendations",
                    insights["recommendations"],
                    bullet_icon="✅",
                )

            st.write("")
            with st.container(border=True):
                st.markdown('<div class="section-heading">📈 Historical Input Comparison</div>', unsafe_allow_html=True)

                if not insights["comparison_table"].empty:

                    st.dataframe(
                        insights["comparison_table"],
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info(
                        "Not enough historical information for comparison."
                    )

                st.write(insights["yield_context"])

                st.caption(
                    "These suggestions are based on historical data, "
                    "not confirmed causes. Soil quality, seeds, irrigation, "
                    "sowing time, pests and farming methods can also "
                    "affect crop yield. Pesticide usage alone cannot "
                    "show whether pest control was effective."
                )

            # --------------------------------------------------
            # PREDICTION DETAILS & CLASSIFICATION RANGE
            # --------------------------------------------------

            det_col1, det_col2 = st.columns([1.15, 1])

            with det_col1:
                with st.container(border=True):
                    st.markdown('<div class="section-heading">📋 Prediction Details</div>', unsafe_allow_html=True)

                    display_data = pd.DataFrame({

                        "Parameter": [
                            "Crop",
                            "Area / Country",
                            "Year",
                            "Rainfall",
                            "Temperature",
                            "Pesticides",
                            "Predicted Yield",
                            "Yield Status"
                        ],

                        "Value": [
                            crop,
                            area,
                            str(int(year)),
                            f"{rainfall:,.2f} mm/year",
                            f"{temperature:.2f} °C",
                            f"{pesticides:,.2f} tonnes",
                            f"{predicted_yield:,.2f} hg/ha",
                            status
                        ]

                    })

                    st.dataframe(
                        display_data,
                        use_container_width=True,
                        hide_index=True
                    )

            with det_col2:
                with st.container(border=True):
                    st.markdown('<div class="section-heading">📈 Crop Yield Classification Range</div>', unsafe_allow_html=True)

                    st.markdown(
                        f"""
                        <div style="display: flex; flex-direction: column; gap: 10px;">
                            <div style="background: #FEF2F2; border: 1px solid #FECACA; border-radius: 10px; padding: 12px 14px;">
                                <div style="font-size: 0.8rem; font-weight: 600; color: #991B1B;">🔴 Poor — Below 25th percentile</div>
                                <div style="font-size: 1.25rem; font-weight: 800; color: #991B1B; margin-top: 2px;">&lt; {thresholds['q25']:,.2f} hg/ha</div>
                            </div>
                            <div style="background: #FFF7ED; border: 1px solid #FED7AA; border-radius: 10px; padding: 12px 14px;">
                                <div style="font-size: 0.8rem; font-weight: 600; color: #9A3412;">🟠 Average — 25th to 50th percentile</div>
                                <div style="font-size: 1.25rem; font-weight: 800; color: #9A3412; margin-top: 2px;">{thresholds['q25']:,.2f} – {thresholds['q50']:,.2f} hg/ha</div>
                            </div>
                            <div style="background: #FEFCE8; border: 1px solid #FDE047; border-radius: 10px; padding: 12px 14px;">
                                <div style="font-size: 0.8rem; font-weight: 600; color: #854D0E;">🟡 Good — 50th to 75th percentile</div>
                                <div style="font-size: 1.25rem; font-weight: 800; color: #854D0E; margin-top: 2px;">{thresholds['q50']:,.2f} – {thresholds['q75']:,.2f} hg/ha</div>
                            </div>
                            <div style="background: #ECFDF5; border: 1px solid #6EE7B7; border-radius: 10px; padding: 12px 14px;">
                                <div style="font-size: 0.8rem; font-weight: 600; color: #14532D;">🟢 Excellent — 75th percentile and above</div>
                                <div style="font-size: 1.25rem; font-weight: 800; color: #14532D; margin-top: 2px;">≥ {thresholds['q75']:,.2f} hg/ha</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        except Exception as error:

            st.error("Prediction failed.")

            st.exception(error)


# ==========================================================
# 15. DATA ANALYSIS PAGE
#     BASED ONLY ON THE LATEST PREDICTION
# ==========================================================

elif page == "📊 Data Analysis":

    render_page_header(
        "📊",
        "Prediction Analysis",
        "Understand your predicted crop yield through historical comparisons and simple explanations.",
        badge="Latest Prediction Insights",
    )

    # CHECK WHETHER A PREDICTION EXISTS

    if "last_prediction" not in st.session_state:

        st.markdown(
            """
            <div class="empty-state-card">
                <div class="empty-state-icon">🌱</div>
                <div class="empty-state-title">No prediction is available yet.</div>
                <div class="empty-state-text">
                    Please visit the Predict Yield page and generate
                    a crop yield prediction first. Your prediction
                    analysis will appear here.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _, btn_center, _ = st.columns([1.2, 1, 1.2])
        with btn_center:
            st.button(
                "🔮 Go to Predict Yield",
                type="primary",
                use_container_width=True,
                on_click=navigate_to,
                args=("🔮 Predict Yield",),
            )

    else:

        result = st.session_state["last_prediction"]

        crop = result["crop"]
        area = result["area"]
        year = result["year"]

        predicted_yield = result["predicted_yield"]

        status = result["status"]

        thresholds = result["thresholds"]

        insights = result["insights"]

        status_style = get_status_style(status)

        # --------------------------------------------------
        # 1. PREDICTION SUMMARY
        # --------------------------------------------------

        st.markdown('<div class="section-heading">🌾 Your Prediction Summary</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_kpi_card("Selected Crop", crop, "Target crop", "🌱", "#10B981")
        with col2:
            render_kpi_card("Selected Area", area, "Country / region", "🌍", "#14532D")
        with col3:
            render_kpi_card("Predicted Yield", f"{predicted_yield:,.2f} hg/ha", f"Prediction Year: {year}", "🌾", "#84CC16")
        with col4:
            st.markdown(
                f"""
                <div class="kpi-card" style="background: {status_style['bg']}; border-color: {status_style['border']}; border-top: 3px solid {status_style['accent']};">
                    <div class="kpi-label" style="color: {status_style['text']};">Predicted Result</div>
                    <div class="kpi-value" style="font-size: 1.4rem; color: {status_style['text']};">{html.escape(status)}</div>
                    <div class="kpi-sub" style="color: {status_style['text']};">Crop-specific tier</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        # --------------------------------------------------
        # 2. HISTORICAL YIELD COMPARISON
        # --------------------------------------------------

        with st.container(border=True):
            st.markdown('<div class="section-heading">📈 How Does Your Prediction Compare?</div>', unsafe_allow_html=True)

            st.write(
                f"""
                The chart below compares your predicted yield
                for **{crop}** with the historical yield levels
                of the same crop.
                """
            )

            comparison = pd.DataFrame({

                "Yield Level": [
                    "Lower historical level",
                    "Historical median",
                    "Historical average",
                    "Upper historical level",
                    "Your prediction"
                ],

                "Yield (hg/ha)": [
                    thresholds["q25"],
                    thresholds["q50"],
                    thresholds["average"],
                    thresholds["q75"],
                    predicted_yield
                ]

            })

            comparison["Category"] = comparison["Yield Level"].apply(
                lambda x: "Your Prediction" if x == "Your prediction" else "Historical Benchmark"
            )

            yield_order = [
                "Lower historical level",
                "Historical median",
                "Historical average",
                "Upper historical level",
                "Your prediction",
            ]

            bars = (
                alt.Chart(comparison)
                .mark_bar(cornerRadiusEnd=6, size=32)
                .encode(
                    y=alt.Y(
                        "Yield Level:N",
                        sort=yield_order,
                        title=None,
                        axis=alt.Axis(labelLimit=200, labelFontSize=12, labelColor="#1F2937"),
                    ),
                    x=alt.X(
                        "Yield (hg/ha):Q",
                        title="Yield (hg/ha)",
                        axis=alt.Axis(format=",.0f", labelColor="#6B7280", titleColor="#1F2937", gridColor="#EEF2EC"),
                    ),
                    color=alt.Color(
                        "Category:N",
                        scale=alt.Scale(
                            domain=["Historical Benchmark", "Your Prediction"],
                            range=["#6EE7B7", "#14532D"],
                        ),
                        legend=alt.Legend(title=None, orient="top"),
                    ),
                    tooltip=[
                        alt.Tooltip("Yield Level:N", title="Level"),
                        alt.Tooltip("Yield (hg/ha):Q", title="Yield (hg/ha)", format=",.2f"),
                    ],
                )
            )

            labels = (
                alt.Chart(comparison)
                .mark_text(align="left", baseline="middle", dx=6, fontSize=12, fontWeight="bold", color="#1F2937")
                .encode(
                    y=alt.Y("Yield Level:N", sort=yield_order),
                    x=alt.X("Yield (hg/ha):Q"),
                    text=alt.Text("Yield (hg/ha):Q", format=",.2f"),
                )
            )

            max_yield_val = max(comparison["Yield (hg/ha)"].max(), 1.0)
            yield_chart = (
                (bars + labels)
                .properties(height=275)
                .encode(x=alt.X("Yield (hg/ha):Q", scale=alt.Scale(domain=[0, max_yield_val * 1.18])))
                .configure_view(strokeWidth=0)
            )

            st.altair_chart(yield_chart, use_container_width=True)

            average = thresholds["average"]

            difference = predicted_yield - average

            if difference > 0:

                st.success(
                    f"Your predicted yield is {difference:,.2f} hg/ha "
                    f"higher than the historical average for {crop}."
                )

            elif difference < 0:

                st.warning(
                    f"Your predicted yield is {abs(difference):,.2f} "
                    f"hg/ha lower than the historical average for {crop}."
                )

            else:

                st.info(
                    "Your predicted yield is equal to the historical average."
                )

            st.caption(
                f"Historical comparison uses "
                f"{thresholds['count']:,} records for {crop}."
            )

        st.write("")

        # --------------------------------------------------
        # 3. INPUT CONDITION ANALYSIS
        # --------------------------------------------------

        st.markdown('<div class="section-heading">🌦️ Analysis of Your Input Conditions</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtext">The following section checks whether your rainfall, temperature and pesticide values are within the usual historical ranges for the selected crop and area, where enough records are available.</div>',
            unsafe_allow_html=True,
        )

        reference, reference_label = get_reference_data(
            crop,
            area
        )

        input_details = [

            (
                "🌧️ Rainfall",
                result["rainfall"],
                "average_rain_fall_mm_per_year",
                "mm/year"
            ),

            (
                "🌡️ Temperature",
                result["temperature"],
                "avg_temp",
                "°C"
            ),

            (
                "🧪 Pesticide Usage",
                result["pesticides"],
                "pesticides_tonnes",
                "tonnes"
            )

        ]

        cond_cols = st.columns(3)

        for idx, (name, user_value, column, unit) in enumerate(input_details):

            historical = pd.to_numeric(
                reference[column],
                errors="coerce"
            ).dropna()

            if historical.empty:
                continue

            q25 = historical.quantile(0.25)
            median_val = historical.median()
            q75 = historical.quantile(0.75)

            with cond_cols[idx % 3]:
                with st.container(border=True):
                    st.markdown(f"**{name}**")

                    m1, m2 = st.columns(2)
                    m1.metric(
                        "Your Input",
                        f"{user_value:,.2f} {unit}"
                    )
                    m2.metric(
                        "Usual Historical Range",
                        f"{q25:,.2f} – {q75:,.2f} {unit}"
                    )

                    range_min = min(float(historical.min()), float(user_value), float(q25))
                    range_max = max(float(historical.max()), float(user_value), float(q75))
                    if range_max == range_min:
                        range_max = range_min + 1.0

                    band_df = pd.DataFrame({
                        "label": ["Range"],
                        "q25": [float(q25)],
                        "q75": [float(q75)],
                        "median": [float(median_val)],
                        "user": [float(user_value)],
                    })

                    base_band = alt.Chart(band_df).encode(y=alt.Y("label:N", axis=None))

                    iqr_bar = base_band.mark_bar(height=14, cornerRadius=4, color="#D1FAE5", stroke="#10B981", strokeWidth=1).encode(
                        x=alt.X(
                            "q25:Q",
                            title=unit,
                            scale=alt.Scale(domain=[range_min * 0.95 if range_min > 0 else range_min, range_max * 1.05]),
                            axis=alt.Axis(labelColor="#6B7280", titleColor="#6B7280", tickCount=4),
                        ),
                        x2="q75:Q",
                        tooltip=[
                            alt.Tooltip("q25:Q", title="Lower (25th %)", format=",.2f"),
                            alt.Tooltip("median:Q", title="Median", format=",.2f"),
                            alt.Tooltip("q75:Q", title="Upper (75th %)", format=",.2f"),
                        ],
                    )

                    median_tick = base_band.mark_tick(color="#10B981", thickness=2.5, size=20).encode(
                        x="median:Q",
                        tooltip=[alt.Tooltip("median:Q", title="Historical Median", format=",.2f")],
                    )

                    user_point = base_band.mark_point(filled=True, size=140, color="#14532D", shape="diamond").encode(
                        x="user:Q",
                        tooltip=[alt.Tooltip("user:Q", title="Your Input", format=",.2f")],
                    )

                    range_chart = (
                        (iqr_bar + median_tick + user_point)
                        .properties(height=68)
                        .configure_view(strokeWidth=0)
                    )

                    st.altair_chart(range_chart, use_container_width=True)

                    if user_value < q25:

                        st.warning(
                            f"{name} is lower than the usual "
                            "historical range."
                        )

                    elif user_value > q75:

                        st.warning(
                            f"{name} is higher than the usual "
                            "historical range."
                        )

                    else:

                        st.success(
                            f"{name} is within the usual "
                            "historical range."
                        )

        st.caption(
            f"Historical comparison: {reference_label} "
            f"({len(reference):,} records)."
        )

        st.write("")

        # --------------------------------------------------
        # 4. POSSIBLE FACTORS & 5. SUGGESTIONS
        # --------------------------------------------------

        fac_col1, fac_col2 = st.columns(2)
        with fac_col1:
            render_insight_list_card(
                "🔍 What May Need Attention?",
                insights["possible_factors"],
                bullet_icon="🔎",
            )
        with fac_col2:
            render_insight_list_card(
                "💡 Suggestions",
                insights["recommendations"],
                bullet_icon="✅",
            )

        st.write("")

        # --------------------------------------------------
        # 6. OVERALL PREDICTION EXPLANATION
        # --------------------------------------------------

        with st.container(border=True):
            st.markdown('<div class="section-heading">📝 Overall Prediction Explanation</div>', unsafe_allow_html=True)

            st.write(
                f"""
                For **{crop}** in **{area}** during **{year}**,
                the model predicts a yield of
                **{predicted_yield:,.2f} hg/ha**.

                The predicted result is classified as **{status}**
                when compared with the historical yield levels
                of the selected crop.

                The input analysis identifies conditions that
                may deserve further attention.
                """
            )

            st.caption(
                "These comparisons show historical patterns, "
                "not confirmed causes of crop yield. Actual yield "
                "can also depend on soil quality, seeds, irrigation, "
                "sowing time, pests and farming methods. "
                "The model does not identify the exact cause "
                "of a predicted result."
            )

        # --------------------------------------------------
        # 7. VIEW PREDICTION INPUTS
        # --------------------------------------------------

        with st.expander("📋 View Your Prediction Details"):

            prediction_details = pd.DataFrame({

                "Parameter": [
                    "Crop",
                    "Area",
                    "Year",
                    "Rainfall",
                    "Temperature",
                    "Pesticides",
                    "Predicted Yield",
                    "Yield Category"
                ],

                "Value": [
                    crop,
                    area,
                    str(year),
                    f"{result['rainfall']:,.2f} mm/year",
                    f"{result['temperature']:.2f} °C",
                    f"{result['pesticides']:,.2f} tonnes",
                    f"{predicted_yield:,.2f} hg/ha",
                    status
                ]

            })

            st.dataframe(
                prediction_details,
                use_container_width=True,
                hide_index=True
            )


# ==========================================================
# 16. MODEL PERFORMANCE PAGE
# ==========================================================

elif page == "🤖 Model Performance":

    render_page_header(
        "🤖",
        "Model Performance",
        "Compare the performance of different Machine Learning models used for crop yield prediction.",
        badge="4-Model Evaluation",
    )

    st.info(
        "Model comparison metrics are computed once and cached. "
        "They reflect a fixed 80/20 train-test split with random_state=42."
    )

    # FEATURES AND TARGET

    X = df[
        [
            "Area",
            "Item",
            "Year",
            "average_rain_fall_mm_per_year",
            "pesticides_tonnes",
            "avg_temp"
        ]
    ]

    y = df["hg/ha_yield"]

    categorical_features = [
        "Area",
        "Item"
    ]

    # TRAIN TEST SPLIT

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    @st.cache_data(show_spinner=False)
    def compute_model_results(_X_train, _X_test, _y_train, _y_test):
        """Train all four comparison models and return metrics. Cached after first run."""

        _models = {

            "Linear Regression":
                LinearRegression(),

            "Decision Tree":
                DecisionTreeRegressor(
                    random_state=42
                ),

            "Random Forest":
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                ),

            "Gradient Boosting":
                GradientBoostingRegressor(
                    random_state=42
                )

        }

        _results = []

        for name, algorithm in _models.items():

            preprocessor = ColumnTransformer(

                transformers=[

                    (
                        "categorical",

                        OneHotEncoder(
                            handle_unknown="ignore"
                        ),

                        categorical_features
                    )

                ],

                remainder="passthrough"
            )

            pipeline = Pipeline([

                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    algorithm
                )

            ])

            pipeline.fit(
                _X_train,
                _y_train
            )

            predictions = pipeline.predict(
                _X_test
            )

            _results.append({

                "Model": name,

                "R²": r2_score(
                    _y_test,
                    predictions
                ),

                "RMSE": np.sqrt(
                    mean_squared_error(
                        _y_test,
                        predictions
                    )
                ),

                "MAE": mean_absolute_error(
                    _y_test,
                    predictions
                )

            })

        return pd.DataFrame(_results)

    with st.spinner("Training and evaluating models (first visit only)..."):
        results_df = compute_model_results(X_train, X_test, y_train, y_test)

    # MODEL ORDER FOR CHART SORTING
    model_order = [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
    ]

    # BEST MODEL

    best_row = results_df.loc[
        results_df["R²"].idxmax()
    ]

    st.markdown('<div class="section-heading">🏆 Best Model on This Test Split</div>', unsafe_allow_html=True)

    st.success(
        f"Best Model: {best_row['Model']}"
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("Best Model", best_row["Model"], "Highest R² score", "🏆", "#14532D")
    with col2:
        render_kpi_card("R² Score", f"{best_row['R²']:.4f}", "Explained variance", "📈", "#10B981")
    with col3:
        render_kpi_card("RMSE", f"{best_row['RMSE']:,.2f}", "Root Mean Squared Error", "📉", "#84CC16")
    with col4:
        render_kpi_card("MAE", f"{best_row['MAE']:,.2f}", "Mean Absolute Error", "📏", "#10B981")

    st.write("")
    st.markdown('<div class="section-heading">🧩 Model Comparison Cards</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtext">Each card summarises one model\'s performance. '
        'R² closer to 1.00 is better. Lower MAE and RMSE mean smaller prediction errors.</div>',
        unsafe_allow_html=True,
    )

    # ── derive per-model metadata from actual results ──────────────────────────

    # Rank by R² descending (1 = best)
    _ranked = results_df.sort_values("R²", ascending=False).reset_index(drop=True)
    _rank_map   = {row["Model"]: idx + 1 for idx, row in _ranked.iterrows()}
    _best_rmse  = results_df["RMSE"].min()
    _best_mae   = results_df["MAE"].min()
    _best_r2    = results_df["R²"].max()

    # Model icons
    _model_icons = {
        "Linear Regression":   "📐",
        "Decision Tree":       "🌳",
        "Random Forest":       "🌲",
        "Gradient Boosting":   "⚡",
    }

    # Model plain-English descriptions
    _model_desc = {
        "Linear Regression":
            "Fits a straight line through the data. Fast and simple, "
            "but limited when relationships are non-linear.",
        "Decision Tree":
            "Splits data into branches using decision rules. "
            "Easy to interpret but can overfit on complex data.",
        "Random Forest":
            "Combines many decision trees and averages their results. "
            "Robust and accurate across varied conditions.",
        "Gradient Boosting":
            "Builds trees sequentially, each correcting the previous. "
            "Often the most accurate but slower to train.",
    }

    def _perf_status(rank, r2):
        if rank == 1:
            return ("🏆 Best Performer",  "#14532D", "#D1FAE5", "#10B981")
        elif r2 >= _best_r2 * 0.92:
            return ("✅ Good Performance", "#166534", "#ECFDF5", "#10B981")
        else:
            return ("⚠️ Needs Improvement", "#92400E", "#FFFBEB", "#F59E0B")

    # ── render 2 × 2 grid ─────────────────────────────────────────────────────

    _rows = [
        list(results_df.iloc[0:2].iterrows()),
        list(results_df.iloc[2:4].iterrows()),
    ]

    for _row_pair in _rows:
        _grid_cols = st.columns(2, gap="medium")
        for _gcol, (_, _r) in zip(_grid_cols, _row_pair):
            _is_best   = _r["Model"] == best_row["Model"]
            _rank      = _rank_map[_r["Model"]]
            _icon      = _model_icons.get(_r["Model"], "🤖")
            _desc      = _model_desc.get(_r["Model"], "")
            _slabel, _stext, _sbg, _sborder = _perf_status(_rank, _r["R²"])

            # R² fill bar — clamp between 0 and 100 %
            _r2_pct    = max(0.0, min(1.0, float(_r["R²"]))) * 100

            # Colour for fill bar segments
            _bar_color = "#10B981" if _is_best else "#6EE7B7"

            # Error indicator arrows
            _rmse_ind  = ('<span style="color:#10B981;font-weight:700;">▼ lowest</span>'
                          if abs(float(_r["RMSE"]) - _best_rmse) < 0.01
                          else '<span style="color:#9CA3AF;">—</span>')
            _mae_ind   = ('<span style="color:#10B981;font-weight:700;">▼ lowest</span>'
                          if abs(float(_r["MAE"]) - _best_mae) < 0.01
                          else '<span style="color:#9CA3AF;">—</span>')

            # Outer card style
            if _is_best:
                _card_style = (
                    "background:linear-gradient(160deg,#ECFDF5 0%,#FFFFFF 55%);"
                    "border:2px solid #10B981;"
                    "border-radius:16px;"
                    "padding:22px 24px;"
                    "box-shadow:0 6px 24px rgba(16,185,129,0.14);"
                    "height:100%;"
                )
            else:
                _card_style = (
                    "background:#FFFFFF;"
                    "border:1px solid #E5E7EB;"
                    "border-radius:16px;"
                    "padding:22px 24px;"
                    "box-shadow:0 1px 4px rgba(17,24,39,0.05);"
                    "height:100%;"
                )

            with _gcol:
                st.markdown(
                    f"""
                    <div style="{_card_style}">
                      <div style="display:flex;align-items:flex-start;
                                  justify-content:space-between;gap:10px;
                                  margin-bottom:14px;">
                        <div style="display:flex;align-items:center;gap:10px;">
                          <div style="width:42px;height:42px;border-radius:11px;
                                      background:{'#D1FAE5' if _is_best else '#F3F4F6'};
                                      border:1px solid {'#A7F3D0' if _is_best else '#E5E7EB'};
                                      display:flex;align-items:center;
                                      justify-content:center;font-size:1.3rem;
                                      flex-shrink:0;">{_icon}</div>
                          <div>
                            <div style="font-family:'Plus Jakarta Sans',sans-serif;
                                        font-size:1.05rem;font-weight:800;
                                        color:#14532D;line-height:1.2;">
                              {html.escape(_r['Model'])}
                            </div>
                            <div style="font-size:0.76rem;color:#6B7280;
                                        margin-top:2px;">Rank #{_rank} of 4</div>
                          </div>
                        </div>
                        <div style="background:{_sbg};border:1px solid {_sborder};
                                    border-radius:999px;padding:4px 11px;
                                    font-size:0.73rem;font-weight:700;
                                    color:{_stext};white-space:nowrap;
                                    flex-shrink:0;">
                          {_slabel}
                        </div>
                      </div>
                      <div style="margin-bottom:16px;">
                        <div style="display:flex;justify-content:space-between;
                                    align-items:baseline;margin-bottom:5px;">
                          <span style="font-size:0.78rem;font-weight:600;color:#6B7280;">
                            R&#178; Score
                            <span style="font-size:0.72rem;font-weight:400;">
                              (higher = better)
                            </span>
                          </span>
                          <span style="font-family:'Plus Jakarta Sans',sans-serif;
                                       font-size:1.3rem;font-weight:800;
                                       color:{'#14532D' if _is_best else '#1F2937'};">
                            {float(_r['R\u00b2']):.4f}
                          </span>
                        </div>
                        <div style="background:#E5E7EB;border-radius:999px;
                                    height:8px;overflow:hidden;">
                          <div style="width:{_r2_pct:.1f}%;height:100%;
                                      background:{'linear-gradient(90deg,#10B981,#84CC16)' if _is_best else _bar_color};
                                      border-radius:999px;">
                          </div>
                        </div>
                        <div style="display:flex;justify-content:space-between;
                                    font-size:0.69rem;color:#9CA3AF;margin-top:3px;">
                          <span>0.00</span><span>0.50</span><span>1.00</span>
                        </div>
                      </div>
                      <div style="border-top:1px solid #F3F4F6;padding-top:12px;">
                        <div style="display:flex;align-items:center;
                                    justify-content:space-between;margin-bottom:8px;">
                          <div>
                            <div style="font-size:0.78rem;font-weight:600;
                                        color:#6B7280;">MAE</div>
                            <div style="font-size:0.68rem;color:#9CA3AF;
                                        margin-top:1px;">Avg error (lower = better)</div>
                          </div>
                          <div style="text-align:right;">
                            <div style="font-size:1.05rem;font-weight:700;color:#1F2937;">
                              {float(_r['MAE']):,.2f}
                              <span style="font-size:0.72rem;font-weight:400;
                                           color:#9CA3AF;">hg/ha</span>
                            </div>
                            <div style="font-size:0.72rem;margin-top:1px;">{_mae_ind}</div>
                          </div>
                        </div>
                        <div style="display:flex;align-items:center;
                                    justify-content:space-between;">
                          <div>
                            <div style="font-size:0.78rem;font-weight:600;
                                        color:#6B7280;">RMSE</div>
                            <div style="font-size:0.68rem;color:#9CA3AF;
                                        margin-top:1px;">Penalises large errors more</div>
                          </div>
                          <div style="text-align:right;">
                            <div style="font-size:1.05rem;font-weight:700;color:#1F2937;">
                              {float(_r['RMSE']):,.2f}
                              <span style="font-size:0.72rem;font-weight:400;
                                           color:#9CA3AF;">hg/ha</span>
                            </div>
                            <div style="font-size:0.72rem;margin-top:1px;">{_rmse_ind}</div>
                          </div>
                        </div>
                      </div>
                      <div style="margin-top:14px;padding:10px 12px;
                                  background:{'#F0FDF4' if _is_best else '#F9FAFB'};
                                  border-radius:9px;
                                  border:1px solid {'#A7F3D0' if _is_best else '#E5E7EB'};
                                  font-size:0.82rem;color:#4B5563;line-height:1.5;">
                        {html.escape(_desc)}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.write("")     # vertical gap between the two rows

    # ── Dynamic interpretation paragraph ──────────────────────────────────────

    _second_best = _ranked.iloc[1]
    _r2_gap      = float(best_row["R²"]) - float(_second_best["R²"])
    _rmse_gap    = float(_second_best["RMSE"]) - float(best_row["RMSE"])

    _gap_desc = (
        f"its R² score is {_r2_gap:.4f} higher"
        if _r2_gap >= 0.001
        else "its R² score is nearly identical to the next best model"
    )
    _error_desc = (
        f"and its RMSE is {_rmse_gap:,.0f} hg/ha lower"
        if _rmse_gap > 0
        else "with comparable error levels"
    )

    st.markdown(
        f"""
        <div style="background:#ECFDF5;border:1px solid #A7F3D0;
                    border-left:4px solid #10B981;border-radius:12px;
                    padding:16px 20px;margin-top:4px;">
          <div style="font-family:'Plus Jakarta Sans',sans-serif;
                      font-size:0.95rem;font-weight:700;color:#14532D;
                      margin-bottom:6px;">
            🏆 Performance Summary
          </div>
          <div style="font-size:0.91rem;color:#1F2937;line-height:1.65;">
            <strong>{html.escape(str(best_row['Model']))}</strong> is the best-performing
            model on this evaluation split.
            Compared with <strong>{html.escape(str(_second_best['Model']))}</strong>
            (the next-closest model), {_gap_desc} {_error_desc}.
            A higher R² means the model explains more of the variation in crop yield.
            Lower MAE and RMSE mean its predictions stay closer to the actual values.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # MODEL COMPARISON TABLE

    with st.container(border=True):
        st.markdown('<div class="section-heading">📊 Model Comparison</div>', unsafe_allow_html=True)

        formatted_results = results_df.copy()

        formatted_results["R²"] = (
            formatted_results["R²"].round(4)
        )

        formatted_results["RMSE"] = (
            formatted_results["RMSE"].round(2)
        )

        formatted_results["MAE"] = (
            formatted_results["MAE"].round(2)
        )

        st.dataframe(
            formatted_results,
            use_container_width=True,
            hide_index=True
        )

    st.write("")

    chart_col1, chart_col2 = st.columns(2)

    # R2 CHART
    with chart_col1:
        with st.container(border=True):
            st.markdown('<div class="section-heading">📈 R² Score Comparison</div>', unsafe_allow_html=True)

            r2_df = results_df[["Model", "R²"]].copy()
            r2_df["Is Best"] = r2_df["Model"].apply(
                lambda m: "Best Model" if m == best_row["Model"] else "Other Models"
            )

            r2_bars = (
                alt.Chart(r2_df)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=44)
                .encode(
                    x=alt.X(
                        "Model:N",
                        sort=model_order,
                        title=None,
                        axis=alt.Axis(labelAngle=-15, labelColor="#1F2937", labelFontSize=12),
                    ),
                    y=alt.Y(
                        "R²:Q",
                        title="R² Score",
                        scale=alt.Scale(domain=[0, 1.08]),
                        axis=alt.Axis(format=".2f", labelColor="#6B7280", titleColor="#1F2937", gridColor="#EEF2EC"),
                    ),
                    color=alt.Color(
                        "Is Best:N",
                        scale=alt.Scale(
                            domain=["Best Model", "Other Models"],
                            range=["#14532D", "#6EE7B7"],
                        ),
                        legend=alt.Legend(title=None, orient="top"),
                    ),
                    tooltip=[
                        alt.Tooltip("Model:N", title="Model"),
                        alt.Tooltip("R²:Q", title="R² Score", format=".4f"),
                    ],
                )
            )

            r2_labels = (
                alt.Chart(r2_df)
                .mark_text(dy=-8, fontSize=12, fontWeight="bold", color="#1F2937")
                .encode(
                    x=alt.X("Model:N", sort=model_order),
                    y=alt.Y("R²:Q"),
                    text=alt.Text("R²:Q", format=".4f"),
                )
            )

            st.altair_chart(
                (r2_bars + r2_labels).properties(height=300).configure_view(strokeWidth=0),
                use_container_width=True,
            )

    # ERROR CHART
    with chart_col2:
        with st.container(border=True):
            st.markdown('<div class="section-heading">📉 Error Comparison</div>', unsafe_allow_html=True)

            error_melted = results_df.melt(
                id_vars="Model",
                value_vars=["RMSE", "MAE"],
                var_name="Metric",
                value_name="Error Value",
            )

            error_chart = (
                alt.Chart(error_melted)
                .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                .encode(
                    x=alt.X(
                        "Model:N",
                        sort=model_order,
                        title=None,
                        axis=alt.Axis(labelAngle=-15, labelColor="#1F2937", labelFontSize=12),
                    ),
                    xOffset=alt.XOffset("Metric:N"),
                    y=alt.Y(
                        "Error Value:Q",
                        title="Error (hg/ha)",
                        axis=alt.Axis(format=",.0f", labelColor="#6B7280", titleColor="#1F2937", gridColor="#EEF2EC"),
                    ),
                    color=alt.Color(
                        "Metric:N",
                        scale=alt.Scale(
                            domain=["RMSE", "MAE"],
                            range=["#14532D", "#10B981"],
                        ),
                        legend=alt.Legend(title=None, orient="top"),
                    ),
                    tooltip=[
                        alt.Tooltip("Model:N", title="Model"),
                        alt.Tooltip("Metric:N", title="Metric"),
                        alt.Tooltip("Error Value:Q", title="Error (hg/ha)", format=",.2f"),
                    ],
                )
                .properties(height=300)
                .configure_view(strokeWidth=0)
            )

            st.altair_chart(error_chart, use_container_width=True)

    st.caption(
        "These metrics describe this particular random test split. "
        "They do not guarantee identical performance on future data."
    )


# ==========================================================
# 17. ABOUT PAGE
# ==========================================================

elif page == "ℹ️ About":

    render_page_header(
        "ℹ️",
        "About the Project",
        "Crop Yield Prediction is a Machine Learning application that predicts crop yield using historical agricultural and environmental information.",
        badge="Project Overview",
    )

    st.markdown('<div class="section-heading">🌱 Input Features</div>', unsafe_allow_html=True)

    feature_cards = [
        ("🌾", "Crop", "Selected crop type."),
        ("🌍", "Area / Country", "Selected geographical area."),
        ("📅", "Year", "Year of prediction."),
        ("🌧️", "Rainfall", "Average annual rainfall in mm/year."),
        ("🌡️", "Temperature", "Average temperature in °C."),
        ("🧪", "Pesticides", "Pesticide usage in tonnes."),
    ]

    f_row1 = st.columns(3)
    for col, (icon, fname, fdesc) in zip(f_row1, feature_cards[:3]):
        with col:
            render_info_card(fname, fdesc, icon)

    st.write("")
    f_row2 = st.columns(3)
    for col, (icon, fname, fdesc) in zip(f_row2, feature_cards[3:]):
        with col:
            render_info_card(fname, fdesc, icon)

    st.write("")

    about_col1, about_col2 = st.columns(2)

    with about_col1:
        st.markdown(
            """
            <div class="agri-card">
                <div class="insight-list-title">🤖 Machine Learning</div>
                <p style="font-size: 0.94rem; color: #374151; line-height: 1.6; margin-bottom: 12px;">
                    The prediction page uses the trained model saved in
                    <code>models/crop_yield_model.pkl</code>.
                </p>
                <p style="font-size: 0.94rem; color: #374151; margin-bottom: 8px;">
                    The Model Performance page compares:
                </p>
                <div class="insight-item"><span>📐</span><span>Linear Regression</span></div>
                <div class="insight-item"><span>🌳</span><span>Decision Tree</span></div>
                <div class="insight-item"><span>🌲</span><span>Random Forest</span></div>
                <div class="insight-item"><span>⚡</span><span>Gradient Boosting</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with about_col2:
        st.markdown(
            """
            <div class="agri-card">
                <div class="insight-list-title">🔍 Prediction Analysis</div>
                <p style="font-size: 0.94rem; color: #374151; line-height: 1.6; margin-bottom: 10px;">
                    The Data Analysis page is based on the latest prediction.
                </p>
                <p style="font-size: 0.94rem; color: #374151; line-height: 1.6; margin-bottom: 10px;">
                    It compares the predicted yield and the entered agricultural conditions with historical records.
                </p>
                <p style="font-size: 0.94rem; color: #374151; line-height: 1.6; margin-bottom: 14px;">
                    It also provides simple explanations and suggestions for further investigation.
                </p>
                <div class="agri-callout" style="margin: 0; font-size: 0.85rem; color: #4B5563;">
                    The analysis identifies historical patterns, not proven causes. The dataset does not include every factor that affects crop yield, such as soil quality, seed variety, irrigation, sowing date, pest pressure and farming practices.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown('<div class="section-heading">📏 Evaluation Metrics</div>', unsafe_allow_html=True)

    met_col1, met_col2, met_col3 = st.columns(3)
    with met_col1:
        render_info_card(
            "R²",
            "Indicates how much variation in yield is explained by the model on evaluation data.",
            "📈",
        )
    with met_col2:
        render_info_card(
            "RMSE",
            "Measures prediction error and gives greater weight to large errors.",
            "📉",
        )
    with met_col3:
        render_info_card(
            "MAE",
            "Measures the average absolute prediction error.",
            "🎯",
        )

    st.write("")
    st.markdown('<div class="section-heading">🗂️ Dataset Summary</div>', unsafe_allow_html=True)

    ds_col1, ds_col2, ds_col3, ds_col4 = st.columns(4)
    with ds_col1:
        render_kpi_card("Crops", df["Item"].nunique(), "Unique crop varieties", "🌾", "#10B981")
    with ds_col2:
        render_kpi_card("Areas", df["Area"].nunique(), "Countries & regions", "🌍", "#14532D")
    with ds_col3:
        render_kpi_card("Records", f"{len(df):,}", "Historical observations", "📊", "#84CC16")
    with ds_col4:
        render_kpi_card("Year Range", f"{int(df['Year'].min())}–{int(df['Year'].max())}", "Data coverage period", "📅", "#10B981")

    st.write("")
    st.markdown('<div class="section-heading">🧭 How to Use This Application</div>', unsafe_allow_html=True)

    steps = [
        ("1️⃣", "Go to Predict Yield", "Select a crop, area, year, and enter the climate conditions. Click Predict Yield to generate a prediction."),
        ("2️⃣", "Review the Result", "The predicted yield value and its quality tier (Poor / Average / Good / Excellent) are shown immediately."),
        ("3️⃣", "Explore Data Analysis", "Visit the Data Analysis page to compare your prediction with historical benchmarks and review input conditions."),
        ("4️⃣", "Check Model Performance", "Visit Model Performance to see how four different ML models compare on R², RMSE and MAE metrics."),
    ]

    step_cols = st.columns(2)
    for i, (num, title, desc) in enumerate(steps):
        with step_cols[i % 2]:
            st.markdown(
                f"""
                <div class="agri-card" style="margin-bottom: 10px;">
                    <div style="display: flex; align-items: flex-start; gap: 12px;">
                        <div style="font-size: 1.5rem; line-height: 1; flex-shrink: 0;">{num}</div>
                        <div>
                            <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.98rem; font-weight: 700; color: #14532D; margin-bottom: 4px;">{html.escape(title)}</div>
                            <div style="font-size: 0.91rem; color: #4B5563; line-height: 1.55;">{html.escape(desc)}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown(
        """
        <div class="agri-callout">
            📂 <strong>Dataset source:</strong> CSV file in the project's <code>data/raw</code> folder.
            Yield values are in <strong>hg/ha</strong> (hectograms per hectare).
            The trained model is stored at <code>models/crop_yield_model.pkl</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================================================
# 18. WHAT-IF ANALYSIS PAGE
# ==========================================================

elif page == "🔀 What-if Analysis":

    render_page_header(
        "🔀",
        "What-if Crop Yield Analysis",
        "Explore how changes in environmental conditions may affect predicted crop yield.",
        badge="Scenario Explorer",
    )

    # ----------------------------------------------------------------
    # STEP 1  Determine whether a baseline prediction already exists.
    # If yes: pre-fill from session state.
    # If no : show a compact form so the user can set crop/area/year.
    # ----------------------------------------------------------------

    _has_baseline = "last_prediction" in st.session_state

    if _has_baseline:
        _base       = st.session_state["last_prediction"]
        _crop       = _base["crop"]
        _area       = _base["area"]
        _year       = int(_base["year"])
        _base_rain  = float(_base["rainfall"])
        _base_temp  = float(_base["temperature"])
        _base_pest  = float(_base["pesticides"])
        _base_yield = float(_base["predicted_yield"])

        st.markdown(
            f"""
            <div class="agri-callout">
                📌 <strong>Baseline loaded from your latest prediction:</strong>
                &nbsp; {html.escape(_crop)} &nbsp;/&nbsp; {html.escape(_area)}
                &nbsp;/&nbsp; {_year}
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            """
            <div class="agri-callout">
                ℹ️ No previous prediction found.
                Choose a crop, area and year below to set the baseline for your scenario.
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown(
                '<div class="section-heading">🌱 Set Baseline Crop &amp; Location</div>',
                unsafe_allow_html=True,
            )
            _wc1, _wc2, _wc3 = st.columns(3)
            with _wc1:
                _crop = st.selectbox("Select Crop", crop_options, key="wi_crop_sel")
            with _wc2:
                _area = st.selectbox("Select Area / Country", area_options,
                                     key="wi_area_sel")
            with _wc3:
                _year = int(st.number_input("Year", min_value=1961,
                                            max_value=2030, value=2025,
                                            step=1, key="wi_year_sel"))

        # Derive defaults from dataset medians
        _base_rain  = float(pd.to_numeric(
            df["average_rain_fall_mm_per_year"], errors="coerce").dropna().median())
        _base_temp  = float(pd.to_numeric(
            df["avg_temp"], errors="coerce").dropna().median())
        _base_pest  = float(pd.to_numeric(
            df["pesticides_tonnes"], errors="coerce").dropna().median())

        # Predict baseline with those defaults
        _bi = pd.DataFrame({
            "Area":                          [_area],
            "Item":                          [_crop],
            "Year":                          [_year],
            "average_rain_fall_mm_per_year": [_base_rain],
            "pesticides_tonnes":             [_base_pest],
            "avg_temp":                      [_base_temp],
        })
        try:
            _bv = model.predict(_bi)
            _base_yield = float(_bv[0])
            if not np.isfinite(_base_yield):
                _base_yield = 0.0
        except Exception:
            _base_yield = 0.0

    # ----------------------------------------------------------------
    # STEP 2  Compute slider bounds from the dataset.
    # ----------------------------------------------------------------

    _wi_crop_df = df[df["Item"] == _crop]
    _wi_ref     = _wi_crop_df if len(_wi_crop_df) >= 5 else df

    def _wi_bounds(col, floor=0.0, headroom=1.4):
        s = pd.to_numeric(_wi_ref[col], errors="coerce").dropna()
        if s.empty:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
        return max(floor, float(s.min())), float(s.max()) * headroom

    _rf_min,   _rf_max   = _wi_bounds("average_rain_fall_mm_per_year", 0.0,  1.4)
    _tmp_min,  _tmp_max  = _wi_bounds("avg_temp",                      -15.0, 1.2)
    _pest_min, _pest_max = _wi_bounds("pesticides_tonnes",              0.0,  2.0)

    # Clamp baseline inside bounds
    _init_rain = float(np.clip(_base_rain, _rf_min,   _rf_max))
    _init_temp = float(np.clip(_base_temp, _tmp_min,  _tmp_max))
    _init_pest = float(np.clip(_base_pest, _pest_min, _pest_max))

    # ----------------------------------------------------------------
    # STEP 3  Persist originals in session state so Reset works.
    # ----------------------------------------------------------------

    _wi_sk = f"wi_orig_{_crop}_{_area}_{_year}"
    if _wi_sk not in st.session_state:
        st.session_state[_wi_sk] = {
            "rain": _init_rain,
            "temp": _init_temp,
            "pest": _init_pest,
        }
    _wi_orig = st.session_state[_wi_sk]

    # Reset button  (must come before sliders so the values take effect)
    _rc, _rsp = st.columns([1, 5])
    with _rc:
        if st.button(
            "↺ Reset to Original",
            type="secondary",
            use_container_width=True,
            key="wi_reset_btn",
        ):
            st.session_state["wi_rain"] = _wi_orig["rain"]
            st.session_state["wi_temp"] = _wi_orig["temp"]
            st.session_state["wi_pest"] = _wi_orig["pest"]
            st.rerun()

    # ----------------------------------------------------------------
    # STEP 4  Sliders
    # ----------------------------------------------------------------

    st.write("")
    with st.container(border=True):
        st.markdown(
            '<div class="section-heading">🎚️ Adjust Environmental Conditions</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-subtext">' +
            "Move the sliders to explore different scenarios. "
            "The existing trained model recalculates the prediction instantly." +
            '</div>',
            unsafe_allow_html=True,
        )

        _ws1, _ws2, _ws3 = st.columns(3)

        with _ws1:
            st.markdown(
                '<div class="input-group-label">🌧️ Rainfall</div>',
                unsafe_allow_html=True,
            )
            _new_rain = st.slider(
                "Average Rainfall mm/year",
                min_value=float(_rf_min),
                max_value=float(_rf_max),
                value=float(st.session_state.get("wi_rain", _init_rain)),
                step=max(1.0, round((_rf_max - _rf_min) / 200, 1)),
                format="%.1f",
                key="wi_rain",
                label_visibility="collapsed",
                help=(
                    f"Dataset range for this crop: "
                    f"{_rf_min:,.0f} to {_rf_max:,.0f} mm/year"
                ),
            )
            st.caption(
                f"Min {_rf_min:,.0f}  |  "
                f"Selected: {_new_rain:,.1f} mm/year  |  "
                f"Max {_rf_max:,.0f}"
            )

        with _ws2:
            st.markdown(
                '<div class="input-group-label">🌡️ Temperature</div>',
                unsafe_allow_html=True,
            )
            _new_temp = st.slider(
                "Average Temperature C",
                min_value=float(_tmp_min),
                max_value=float(_tmp_max),
                value=float(st.session_state.get("wi_temp", _init_temp)),
                step=0.5,
                format="%.1f",
                key="wi_temp",
                label_visibility="collapsed",
                help=(
                    f"Dataset range for this crop: "
                    f"{_tmp_min:.1f} to {_tmp_max:.1f} C"
                ),
            )
            st.caption(
                f"Min {_tmp_min:.1f}  |  "
                f"Selected: {_new_temp:.1f} C  |  "
                f"Max {_tmp_max:.1f}"
            )

        with _ws3:
            st.markdown(
                '<div class="input-group-label">🧪 Pesticides</div>',
                unsafe_allow_html=True,
            )
            _new_pest = st.slider(
                "Pesticides Usage tonnes",
                min_value=float(_pest_min),
                max_value=float(_pest_max),
                value=float(st.session_state.get("wi_pest", _init_pest)),
                step=max(1.0, round((_pest_max - _pest_min) / 200, 1)),
                format="%.1f",
                key="wi_pest",
                label_visibility="collapsed",
                help=(
                    f"Dataset range for this crop: "
                    f"{_pest_min:,.0f} to {_pest_max:,.0f} tonnes"
                ),
            )
            st.caption(
                f"Min {_pest_min:,.0f}  |  "
                f"Selected: {_new_pest:,.1f} t  |  "
                f"Max {_pest_max:,.0f}"
            )

    # ----------------------------------------------------------------
    # STEP 5  Run the existing model on the modified scenario.
    #         Uses the loaded `model` object  no retraining.
    # ----------------------------------------------------------------

    _wi_input = pd.DataFrame({
        "Area":                          [_area],
        "Item":                          [_crop],
        "Year":                          [_year],
        "average_rain_fall_mm_per_year": [_new_rain],
        "pesticides_tonnes":             [_new_pest],
        "avg_temp":                      [_new_temp],
    })

    try:
        _wi_pred   = model.predict(_wi_input)
        _new_yield = float(_wi_pred[0])
        if not np.isfinite(_new_yield):
            _new_yield = 0.0
    except Exception as _wi_err:
        st.error("Scenario prediction failed.")
        st.exception(_wi_err)
        st.stop()

    _wi_diff  = _new_yield - _base_yield
    _wi_pct   = (_wi_diff / _base_yield * 100) if _base_yield != 0 else 0.0
    _wi_sign  = "+" if _wi_diff >= 0 else ""

    # ----------------------------------------------------------------
    # STEP 6  KPI comparison strip
    # ----------------------------------------------------------------

    st.write("")
    st.markdown(
        '<div class="section-heading">📊 Prediction Comparison</div>',
        unsafe_allow_html=True,
    )

    _wi_thresh       = get_crop_thresholds(_crop)
    _wi_base_st, _   = classify_yield(_base_yield, _wi_thresh)
    _wi_new_st,  _   = classify_yield(_new_yield,  _wi_thresh)
    _wi_base_sty     = get_status_style(_wi_base_st)
    _wi_new_sty      = get_status_style(_wi_new_st)
    _wi_diff_accent  = "#10B981" if _wi_diff >= 0 else "#DC2626"
    _wi_arrow        = "▲" if _wi_diff > 0 else ("▼" if _wi_diff < 0 else "→")

    _wk1, _wk2, _wk3, _wk4 = st.columns(4)

    with _wk1:
        render_kpi_card(
            "Original Yield",
            f"{_base_yield:,.2f}",
            "hg/ha  (baseline)",
            "📌",
            "#6B7280",
        )

    with _wk2:
        st.markdown(
            f"""
            <div class="kpi-card"
                 style="border-top-color:{_wi_new_sty['accent']};
                        background:{_wi_new_sty['bg']};
                        border-color:{_wi_new_sty['border']};">
              <div class="kpi-label" style="color:{_wi_new_sty['text']};">
                🎯 Scenario Yield
              </div>
              <div class="kpi-value" style="color:{_wi_new_sty['text']};">
                {_new_yield:,.2f}
              </div>
              <div class="kpi-sub" style="color:{_wi_new_sty['text']};">hg/ha</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with _wk3:
        render_kpi_card(
            f"{_wi_arrow} Difference",
            f"{_wi_sign}{_wi_diff:,.2f}",
            "hg/ha",
            "",
            _wi_diff_accent,
        )

    with _wk4:
        render_kpi_card(
            "% Change",
            f"{_wi_sign}{_wi_pct:.1f}%",
            "relative to original",
            "📈" if _wi_diff >= 0 else "📉",
            _wi_diff_accent,
        )

    # ----------------------------------------------------------------
    # STEP 7  Side-by-side input comparison table
    # ----------------------------------------------------------------

    st.write("")
    with st.container(border=True):
        st.markdown(
            '<div class="section-heading">🔢 Original vs Modified Conditions</div>',
            unsafe_allow_html=True,
        )
        _wi_comp = pd.DataFrame({
            "Condition": [
                "Rainfall (mm/year)",
                "Temperature (C)",
                "Pesticides (tonnes)",
            ],
            "Original": [
                f"{_init_rain:,.2f}",
                f"{_init_temp:.2f}",
                f"{_init_pest:,.2f}",
            ],
            "Scenario": [
                f"{_new_rain:,.2f}",
                f"{_new_temp:.2f}",
                f"{_new_pest:,.2f}",
            ],
            "Change": [
                f"{_new_rain - _init_rain:+.2f}",
                f"{_new_temp - _init_temp:+.2f}",
                f"{_new_pest - _init_pest:+.2f}",
            ],
        })
        st.dataframe(_wi_comp, use_container_width=True, hide_index=True)

    # ----------------------------------------------------------------
    # STEP 8  Bar chart
    # ----------------------------------------------------------------

    st.write("")
    with st.container(border=True):
        st.markdown(
            '<div class="section-heading">📈 Yield Comparison Chart</div>',
            unsafe_allow_html=True,
        )

        _wi_chart = pd.DataFrame({
            "Scenario":      ["Original Prediction", "Scenario Prediction"],
            "Yield (hg/ha)": [_base_yield, _new_yield],
            "Type":          ["Original", "Scenario"],
        })

        _wi_ymax   = max(_base_yield, _new_yield, 1.0) * 1.22
        _wi_colors = (
            ["#6EE7B7", "#14532D"]
            if _wi_diff >= 0
            else ["#6EE7B7", "#F87171"]
        )

        _wi_bars = (
            alt.Chart(_wi_chart)
            .mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7, size=56)
            .encode(
                x=alt.X(
                    "Scenario:N",
                    title=None,
                    axis=alt.Axis(
                        labelColor="#1F2937",
                        labelFontSize=13,
                        labelFontWeight="bold",
                    ),
                ),
                y=alt.Y(
                    "Yield (hg/ha):Q",
                    title="Yield (hg/ha)",
                    scale=alt.Scale(domain=[0, _wi_ymax]),
                    axis=alt.Axis(
                        format=",.0f",
                        labelColor="#6B7280",
                        titleColor="#1F2937",
                        gridColor="#EEF2EC",
                    ),
                ),
                color=alt.Color(
                    "Type:N",
                    scale=alt.Scale(
                        domain=["Original", "Scenario"],
                        range=_wi_colors,
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Scenario:N", title="Scenario"),
                    alt.Tooltip(
                        "Yield (hg/ha):Q",
                        title="Yield (hg/ha)",
                        format=",.2f",
                    ),
                ],
            )
        )

        _wi_lbl = (
            alt.Chart(_wi_chart)
            .mark_text(dy=-12, fontSize=14, fontWeight="bold", color="#1F2937")
            .encode(
                x=alt.X("Scenario:N"),
                y=alt.Y("Yield (hg/ha):Q"),
                text=alt.Text("Yield (hg/ha):Q", format=",.2f"),
            )
        )

        st.altair_chart(
            (_wi_bars + _wi_lbl)
            .properties(height=330)
            .configure_view(strokeWidth=0),
            use_container_width=True,
        )

    # ----------------------------------------------------------------
    # STEP 9  Yield category comparison cards
    # ----------------------------------------------------------------

    st.write("")
    _wcc1, _wcc2 = st.columns(2)

    with _wcc1:
        st.markdown(
            f"""
            <div style="background:{_wi_base_sty['bg']};
                        border:1.5px solid {_wi_base_sty['border']};
                        border-radius:14px;padding:18px 20px;">
              <div style="font-size:.8rem;font-weight:600;
                          color:{_wi_base_sty['text']};margin-bottom:4px;">
                📌 Original Yield Category
              </div>
              <div style="font-size:1.5rem;font-weight:800;
                          color:{_wi_base_sty['text']};">
                {html.escape(_wi_base_st)}
              </div>
              <div style="font-size:.88rem;color:{_wi_base_sty['text']};margin-top:4px;">
                {_base_yield:,.2f} hg/ha
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with _wcc2:
        st.markdown(
            f"""
            <div style="background:{_wi_new_sty['bg']};
                        border:1.5px solid {_wi_new_sty['border']};
                        border-radius:14px;padding:18px 20px;">
              <div style="font-size:.8rem;font-weight:600;
                          color:{_wi_new_sty['text']};margin-bottom:4px;">
                🎯 Scenario Yield Category
              </div>
              <div style="font-size:1.5rem;font-weight:800;
                          color:{_wi_new_sty['text']};">
                {html.escape(_wi_new_st)}
              </div>
              <div style="font-size:.88rem;color:{_wi_new_sty['text']};margin-top:4px;">
                {_new_yield:,.2f} hg/ha
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ----------------------------------------------------------------
    # STEP 10  Smart explanation  generated from actual results
    # ----------------------------------------------------------------

    st.write("")

    _wi_rain_d = abs(_new_rain - _init_rain)
    _wi_temp_d = abs(_new_temp - _init_temp)
    _wi_pest_d = abs(_new_pest - _init_pest)
    _wi_changed = (_wi_rain_d + _wi_temp_d + _wi_pest_d) > 0.05

    if not _wi_changed:
        _wi_expl = (
            "The slider values match the original conditions. "
            "Move a slider to explore a different scenario."
        )
    else:
        _wi_parts = []
        if _wi_rain_d > 0.5:
            _d = "increased" if _new_rain > _init_rain else "decreased"
            _wi_parts.append(f"rainfall {_d} to {_new_rain:,.1f} mm/year")
        if _wi_temp_d > 0.1:
            _d = "increased" if _new_temp > _init_temp else "decreased"
            _wi_parts.append(f"temperature {_d} to {_new_temp:.1f} C")
        if _wi_pest_d > 0.5:
            _d = "increased" if _new_pest > _init_pest else "decreased"
            _wi_parts.append(f"pesticide usage {_d} to {_new_pest:,.1f} tonnes")

        if not _wi_parts:
            _wi_parts = ["one or more conditions changed"]

        if len(_wi_parts) > 1:
            _wi_change_str = ", ".join(_wi_parts[:-1]) + " and " + _wi_parts[-1]
        else:
            _wi_change_str = _wi_parts[0]

        if abs(_wi_diff) < 1.0:
            _wi_outcome = (
                f"the predicted yield remained almost the same "
                f"({_new_yield:,.2f} hg/ha, difference less than 1 hg/ha)."
            )
        elif _wi_diff > 0:
            _wi_outcome = (
                f"the model predicts a higher yield of {_new_yield:,.2f} hg/ha "
                f"(+{_wi_diff:,.2f} hg/ha, {_wi_pct:.1f}% increase)."
            )
        else:
            _wi_outcome = (
                f"the model predicts a lower yield of {_new_yield:,.2f} hg/ha "
                f"({_wi_diff:,.2f} hg/ha, {_wi_pct:.1f}% decrease)."
            )

        _wi_expl = f"After {_wi_change_str}, {_wi_outcome}"

    st.markdown(
        f"""
        <div class="agri-card" style="border-left:4px solid #10B981;">
          <div style="font-size:.8rem;font-weight:700;color:#14532D;
                      text-transform:uppercase;letter-spacing:.05em;
                      margin-bottom:8px;">
            💬 Scenario Explanation
          </div>
          <div style="font-size:.97rem;color:#1F2937;line-height:1.6;">
            {html.escape(_wi_expl)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------------------
    # STEP 11  Disclaimer
    # ----------------------------------------------------------------

    st.markdown(
        """
        <div class="agri-callout">
            ℹ️ <strong>Important:</strong>
            This page shows model-based scenario results.
            The numbers reflect how the trained ML model responds to the
            adjusted input values. They are not proof that changing rainfall,
            temperature or pesticide usage will directly cause a particular
            yield change in reality.
            Actual crop yield depends on many factors not captured in this
            model, including soil quality, seed variety, irrigation, sowing
            time and farming practices.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# 19. ESTIMATED PROFIT CALCULATOR PAGE
# ==========================================================

elif page == "💰 Profit Calculator":

    render_page_header(
        "💰",
        "Estimated Profit Calculator",
        "Convert your predicted crop yield into an estimated farming revenue, expenses and profit.",
        badge="Financial Estimates",
    )

    # ----------------------------------------------------------------
    # UNIT CONVERSION NOTE
    # The existing model predicts yield in hg/ha (hectograms per hectare).
    # 1 hg = 0.1 kg  =>  1 hg/ha = 0.1 kg/ha
    # Total production (kg) = predicted_yield (hg/ha) * land_area (ha) * 0.1
    # We surface this as a first-class choice so the user always knows
    # exactly what units are flowing through the calculation.
    # ----------------------------------------------------------------

    _HG_TO_KG = 0.1          # 1 hg = 0.1 kg
    _HG_TO_TONNE = 0.0001    # 1 hg = 0.0001 tonne  (100 hg = 1 kg, 1000 kg = 1 tonne)

    # ----------------------------------------------------------------
    # STEP 1  Fetch crop & yield from previous prediction if available
    # ----------------------------------------------------------------

    _pc_has_pred = "last_prediction" in st.session_state

    if _pc_has_pred:
        _pc_pred   = st.session_state["last_prediction"]
        _pc_crop   = _pc_pred["crop"]
        _pc_area   = _pc_pred["area"]
        _pc_yield  = float(_pc_pred["predicted_yield"])   # hg/ha

        st.markdown(
            f"""
            <div class="agri-callout">
                📌 <strong>Prediction loaded:</strong>
                &nbsp; {html.escape(_pc_crop)} &nbsp;/&nbsp; {html.escape(_pc_area)}
                &nbsp;&mdash;&nbsp; Predicted yield: <strong>{_pc_yield:,.2f} hg/ha</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="agri-callout">
                ℹ️ No prediction found. Go to Predict Yield first to load crop data
                automatically, or enter the values manually below.
            </div>
            """,
            unsafe_allow_html=True,
        )
        _pc_crop  = "Unknown crop"
        _pc_area  = "Unknown area"
        _pc_yield = 0.0

    # ----------------------------------------------------------------
    # STEP 2  Production inputs
    # ----------------------------------------------------------------

    st.write("")
    with st.container(border=True):
        st.markdown(
            '<div class="section-heading">🌾 Production Details</div>',
            unsafe_allow_html=True,
        )

        _pi1, _pi2, _pi3 = st.columns(3)

        with _pi1:
            _pc_land = st.number_input(
                "Cultivated Land Area (hectares)",
                min_value=0.01,
                max_value=100000.0,
                value=1.0,
                step=0.5,
                format="%.2f",
                key="pc_land",
                help="Enter the total cultivated area in hectares (1 hectare = 10,000 m²).",
            )

        with _pi2:
            _pc_yield_input = st.number_input(
                "Predicted Yield (hg/ha)",
                min_value=0.0,
                value=float(max(_pc_yield, 0.0)),
                step=100.0,
                format="%.2f",
                key="pc_yield_val",
                help="Loaded from latest prediction. You can edit this manually.",
            )

        with _pi3:
            _pc_harvest_pct = st.slider(
                "Expected Harvest % of Predicted",
                min_value=10,
                max_value=100,
                value=80,
                step=5,
                key="pc_harvest_pct",
                help=(
                    "Adjust for crop losses due to weather, pests or other factors. "
                    "100% means the full predicted yield is harvested."
                ),
            )
            st.caption(
                f"Using {_pc_harvest_pct}% of predicted yield "
                f"= {_pc_yield_input * _pc_harvest_pct / 100:,.2f} hg/ha effective yield."
            )

    # ----------------------------------------------------------------
    # STEP 3  Selling price inputs
    # ----------------------------------------------------------------

    _pc_eff_yield_hgha = _pc_yield_input * (_pc_harvest_pct / 100.0)

    # Total production in kg  (hg/ha * ha * 0.1 kg/hg)
    _pc_prod_kg = _pc_eff_yield_hgha * _pc_land * _HG_TO_KG

    st.write("")
    with st.container(border=True):
        st.markdown(
            '<div class="section-heading">💵 Selling Price</div>',
            unsafe_allow_html=True,
        )

        _sp1, _sp2, _sp3 = st.columns(3)

        with _sp1:
            _pc_currency = st.selectbox(
                "Currency",
                ["USD ($)", "EUR (€)", "GBP (£)", "INR (₹)", "Other"],
                key="pc_currency",
            )
            _pc_curr_sym = _pc_currency.split(" ")[0] if _pc_currency != "Other" else ""

        with _sp2:
            _pc_price_unit = st.selectbox(
                "Price Unit",
                ["per kg", "per tonne (1,000 kg)", "per quintal (100 kg)"],
                key="pc_price_unit",
                help="Choose the unit that matches your local market price.",
            )

        with _sp3:
            _pc_price = st.number_input(
                f"Selling Price ({_pc_price_unit})",
                min_value=0.0,
                value=0.50,
                step=0.01,
                format="%.4f",
                key="pc_price_val",
            )

        # Convert selling price to per-kg regardless of chosen unit
        if "per tonne" in _pc_price_unit:
            _pc_price_per_kg = _pc_price / 1000.0
        elif "per quintal" in _pc_price_unit:
            _pc_price_per_kg = _pc_price / 100.0
        else:                            # per kg
            _pc_price_per_kg = _pc_price

        st.caption(
            f"Effective price per kg: "
            f"{_pc_curr_sym} {_pc_price_per_kg:.4f}  |  "
            f"Total production: {_pc_prod_kg:,.2f} kg  "
            f"({_pc_prod_kg / 1000:,.4f} tonnes)"
        )

    # ----------------------------------------------------------------
    # STEP 4  Expense inputs
    # ----------------------------------------------------------------

    st.write("")
    with st.container(border=True):
        st.markdown(
            '<div class="section-heading">📋 Farming Expenses</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-subtext">Enter total expenses for the cultivated area. Leave at 0 if not applicable.</div>',
            unsafe_allow_html=True,
        )

        _ea, _eb = st.columns(2)
        _ec, _ed = st.columns(2)
        _ee, _ef = st.columns(2)
        _eg, _eh = st.columns(2)

        with _ea:
            _pc_seed = st.number_input(
                f"Seed Expenses ({_pc_curr_sym})",
                min_value=0.0, value=0.0, step=1.0,
                format="%.2f", key="pc_seed",
            )
        with _eb:
            _pc_fert = st.number_input(
                f"Fertilizer Expenses ({_pc_curr_sym})",
                min_value=0.0, value=0.0, step=1.0,
                format="%.2f", key="pc_fert",
            )
        with _ec:
            _pc_pest_exp = st.number_input(
                f"Pesticide Expenses ({_pc_curr_sym})",
                min_value=0.0, value=0.0, step=1.0,
                format="%.2f", key="pc_pest_exp",
            )
        with _ed:
            _pc_labour = st.number_input(
                f"Labour Expenses ({_pc_curr_sym})",
                min_value=0.0, value=0.0, step=1.0,
                format="%.2f", key="pc_labour",
            )
        with _ee:
            _pc_irrig = st.number_input(
                f"Irrigation Expenses ({_pc_curr_sym})",
                min_value=0.0, value=0.0, step=1.0,
                format="%.2f", key="pc_irrig",
            )
        with _ef:
            _pc_other = st.number_input(
                f"Other Expenses ({_pc_curr_sym})",
                min_value=0.0, value=0.0, step=1.0,
                format="%.2f", key="pc_other",
            )

    # ----------------------------------------------------------------
    # STEP 5  Core calculations
    # ----------------------------------------------------------------

    _pc_total_expenses = (
        _pc_seed + _pc_fert + _pc_pest_exp +
        _pc_labour + _pc_irrig + _pc_other
    )
    _pc_revenue = _pc_prod_kg * _pc_price_per_kg
    _pc_profit  = _pc_revenue - _pc_total_expenses
    _pc_margin  = (
        (_pc_profit / _pc_revenue * 100) if _pc_revenue > 0 else 0.0
    )
    _pc_is_profit = _pc_profit >= 0

    # ----------------------------------------------------------------
    # STEP 6  Results KPI strip
    # ----------------------------------------------------------------

    st.write("")
    st.markdown(
        '<div class="section-heading">📊 Financial Summary</div>',
        unsafe_allow_html=True,
    )

    _pk1, _pk2, _pk3, _pk4, _pk5 = st.columns(5)

    with _pk1:
        render_kpi_card(
            "Total Production",
            f"{_pc_prod_kg:,.1f} kg",
            f"{_pc_prod_kg / 1000:,.4f} tonnes",
            "🌾",
            "#10B981",
        )

    with _pk2:
        render_kpi_card(
            "Estimated Revenue",
            f"{_pc_curr_sym} {_pc_revenue:,.2f}",
            f"At {_pc_curr_sym} {_pc_price_per_kg:.4f} / kg",
            "💰",
            "#14532D",
        )

    with _pk3:
        render_kpi_card(
            "Total Expenses",
            f"{_pc_curr_sym} {_pc_total_expenses:,.2f}",
            "All farming costs combined",
            "📋",
            "#84CC16",
        )

    with _pk4:
        _profit_accent = "#10B981" if _pc_is_profit else "#DC2626"
        _profit_icon   = "📈" if _pc_is_profit else "📉"
        _profit_label  = "Estimated Profit" if _pc_is_profit else "Estimated Loss"
        render_kpi_card(
            _profit_label,
            f"{_pc_curr_sym} {abs(_pc_profit):,.2f}",
            "Revenue minus expenses",
            _profit_icon,
            _profit_accent,
        )

    with _pk5:
        _margin_accent = "#10B981" if _pc_margin >= 0 else "#DC2626"
        render_kpi_card(
            "Profit Margin",
            f"{_pc_margin:.1f}%",
            "Of estimated revenue",
            "💯",
            _margin_accent,
        )

    # ----------------------------------------------------------------
    # STEP 7  Colour-coded result card
    # ----------------------------------------------------------------

    st.write("")
    if _pc_revenue == 0:
        st.info(
            "ℹ️ Enter a selling price and land area to see revenue and profit estimates."
        )
    elif _pc_is_profit:
        st.success(
            f"✅ Based on these estimates, the crop yields an estimated profit of "
            f"**{_pc_curr_sym} {_pc_profit:,.2f}** "
            f"on a revenue of **{_pc_curr_sym} {_pc_revenue:,.2f}** "
            f"(profit margin: **{_pc_margin:.1f}%**)."
        )
    else:
        st.warning(
            f"⚠️ Based on these estimates, the expenses exceed the revenue by "
            f"**{_pc_curr_sym} {abs(_pc_profit):,.2f}**. "
            f"Review your costs or adjust the selling price."
        )

    # ----------------------------------------------------------------
    # STEP 8  Revenue vs Expenses chart
    # ----------------------------------------------------------------

    st.write("")
    _chart_col1, _chart_col2 = st.columns(2)

    with _chart_col1:
        with st.container(border=True):
            st.markdown(
                '<div class="section-heading">📈 Revenue vs Expenses</div>',
                unsafe_allow_html=True,
            )

            _rv_df = pd.DataFrame({
                "Category": ["Estimated Revenue", "Total Expenses",
                              "Estimated Profit" if _pc_is_profit else "Estimated Loss"],
                "Amount":   [_pc_revenue, _pc_total_expenses, abs(_pc_profit)],
                "Type":     ["Revenue", "Expenses",
                              "Profit" if _pc_is_profit else "Loss"],
            })

            _rv_colors = {
                "Revenue":  "#6EE7B7",
                "Expenses": "#FCD34D",
                "Profit":   "#14532D",
                "Loss":     "#F87171",
            }

            _rv_chart = (
                alt.Chart(_rv_df)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=44)
                .encode(
                    x=alt.X(
                        "Category:N",
                        title=None,
                        axis=alt.Axis(
                            labelColor="#1F2937",
                            labelFontSize=11,
                            labelAngle=-10,
                        ),
                    ),
                    y=alt.Y(
                        "Amount:Q",
                        title=f"Amount ({_pc_curr_sym})",
                        axis=alt.Axis(
                            format=",.0f",
                            labelColor="#6B7280",
                            titleColor="#1F2937",
                            gridColor="#EEF2EC",
                        ),
                    ),
                    color=alt.Color(
                        "Type:N",
                        scale=alt.Scale(
                            domain=list(_rv_colors.keys()),
                            range=list(_rv_colors.values()),
                        ),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("Category:N", title="Category"),
                        alt.Tooltip("Amount:Q",   title=f"Amount ({_pc_curr_sym})", format=",.2f"),
                    ],
                )
            )
            _rv_lbl = (
                alt.Chart(_rv_df)
                .mark_text(dy=-10, fontSize=11, fontWeight="bold", color="#1F2937")
                .encode(
                    x=alt.X("Category:N"),
                    y=alt.Y("Amount:Q"),
                    text=alt.Text("Amount:Q", format=",.2f"),
                )
            )
            st.altair_chart(
                (_rv_chart + _rv_lbl)
                .properties(height=300)
                .configure_view(strokeWidth=0),
                use_container_width=True,
            )

    # ----------------------------------------------------------------
    # STEP 9  Expense breakdown chart
    # ----------------------------------------------------------------

    with _chart_col2:
        with st.container(border=True):
            st.markdown(
                '<div class="section-heading">🥧 Expense Breakdown</div>',
                unsafe_allow_html=True,
            )

            _exp_items = {
                "Seeds":        _pc_seed,
                "Fertilizer":   _pc_fert,
                "Pesticides":   _pc_pest_exp,
                "Labour":       _pc_labour,
                "Irrigation":   _pc_irrig,
                "Other":        _pc_other,
            }
            _exp_nonzero = {k: v for k, v in _exp_items.items() if v > 0}

            if not _exp_nonzero:
                st.info("Enter expenses above to see the breakdown chart.")
            else:
                _exp_df = pd.DataFrame({
                    "Expense":  list(_exp_nonzero.keys()),
                    "Amount":   list(_exp_nonzero.values()),
                })

                _exp_pie = (
                    alt.Chart(_exp_df)
                    .mark_arc(innerRadius=55, outerRadius=110)
                    .encode(
                        theta=alt.Theta("Amount:Q"),
                        color=alt.Color(
                            "Expense:N",
                            scale=alt.Scale(scheme="greens"),
                            legend=alt.Legend(title=None, orient="bottom"),
                        ),
                        tooltip=[
                            alt.Tooltip("Expense:N", title="Expense"),
                            alt.Tooltip("Amount:Q",  title=f"Amount ({_pc_curr_sym})", format=",.2f"),
                        ],
                    )
                )
                st.altair_chart(
                    _exp_pie
                    .properties(height=300)
                    .configure_view(strokeWidth=0),
                    use_container_width=True,
                )

    # ----------------------------------------------------------------
    # STEP 10  What-if Profit Analysis
    # ----------------------------------------------------------------

    st.write("")
    with st.container(border=True):
        st.markdown(
            '<div class="section-heading">🔀 What-if Profit Analysis</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-subtext">' +
            "Adjust selling price or a key expense to see how the profit changes." +
            '</div>',
            unsafe_allow_html=True,
        )

        _wf1, _wf2 = st.columns(2)

        with _wf1:
            # Selling price slider - range ±50% around current price
            _pc_price_lo  = max(0.0001, _pc_price * 0.5)
            _pc_price_hi  = max(_pc_price * 1.5, _pc_price + 1.0)
            _wi_price = st.slider(
                f"Adjusted Selling Price ({_pc_price_unit})",
                min_value=float(_pc_price_lo),
                max_value=float(_pc_price_hi),
                value=float(_pc_price),
                step=float(max(0.0001, (_pc_price_hi - _pc_price_lo) / 200)),
                format="%.4f",
                key="pc_wi_price",
            )

        with _wf2:
            # Labour slider  - most common variable cost
            _pc_lab_lo = 0.0
            _pc_lab_hi = max(_pc_labour * 2.0, _pc_labour + 100.0)
            _wi_labour = st.slider(
                f"Adjusted Labour Expenses ({_pc_curr_sym})",
                min_value=float(_pc_lab_lo),
                max_value=float(_pc_lab_hi),
                value=float(_pc_labour),
                step=float(max(1.0, (_pc_lab_hi - _pc_lab_lo) / 200)),
                format="%.2f",
                key="pc_wi_labour",
            )

        # Recalculate with adjusted values
        if "per tonne" in _pc_price_unit:
            _wi_price_per_kg = _wi_price / 1000.0
        elif "per quintal" in _pc_price_unit:
            _wi_price_per_kg = _wi_price / 100.0
        else:
            _wi_price_per_kg = _wi_price

        _wi_total_exp = (
            _pc_seed + _pc_fert + _pc_pest_exp +
            _wi_labour + _pc_irrig + _pc_other
        )
        _wi_revenue = _pc_prod_kg * _wi_price_per_kg
        _wi_profit  = _wi_revenue - _wi_total_exp
        _wi_margin  = (_wi_profit / _wi_revenue * 100) if _wi_revenue > 0 else 0.0
        _wi_is_p    = _wi_profit >= 0

        _wk1, _wk2, _wk3, _wk4 = st.columns(4)
        with _wk1:
            render_kpi_card(
                "Scenario Revenue",
                f"{_pc_curr_sym} {_wi_revenue:,.2f}",
                "With adjusted price",
                "💵",
                "#10B981",
            )
        with _wk2:
            render_kpi_card(
                "Scenario Expenses",
                f"{_pc_curr_sym} {_wi_total_exp:,.2f}",
                "With adjusted labour",
                "📋",
                "#84CC16",
            )
        with _wk3:
            _wi_accent = "#10B981" if _wi_is_p else "#DC2626"
            render_kpi_card(
                "Scenario Profit" if _wi_is_p else "Scenario Loss",
                f"{_pc_curr_sym} {abs(_wi_profit):,.2f}",
                "Revenue minus expenses",
                "📈" if _wi_is_p else "📉",
                _wi_accent,
            )
        with _wk4:
            render_kpi_card(
                "Scenario Margin",
                f"{_wi_margin:.1f}%",
                "Of scenario revenue",
                "💯",
                "#10B981" if _wi_margin >= 0 else "#DC2626",
            )

    # ----------------------------------------------------------------
    # STEP 11  Detailed calculation summary table
    # ----------------------------------------------------------------

    st.write("")
    with st.expander("📋 View Full Calculation Breakdown"):
        _summary_df = pd.DataFrame({
            "Item": [
                "Predicted Yield",
                "Harvest % Applied",
                "Effective Yield (after harvest %)",
                "Cultivated Land Area",
                "Total Production (kg)",
                "Total Production (tonnes)",
                "Selling Price per kg",
                "Estimated Revenue",
                "Seed Expenses",
                "Fertilizer Expenses",
                "Pesticide Expenses",
                "Labour Expenses",
                "Irrigation Expenses",
                "Other Expenses",
                "Total Expenses",
                "Estimated Profit / Loss",
                "Profit Margin",
            ],
            "Value": [
                f"{_pc_yield_input:,.2f} hg/ha",
                f"{_pc_harvest_pct}%",
                f"{_pc_eff_yield_hgha:,.2f} hg/ha",
                f"{_pc_land:,.2f} ha",
                f"{_pc_prod_kg:,.4f} kg",
                f"{_pc_prod_kg / 1000:,.6f} tonnes",
                f"{_pc_curr_sym} {_pc_price_per_kg:.4f} / kg",
                f"{_pc_curr_sym} {_pc_revenue:,.2f}",
                f"{_pc_curr_sym} {_pc_seed:,.2f}",
                f"{_pc_curr_sym} {_pc_fert:,.2f}",
                f"{_pc_curr_sym} {_pc_pest_exp:,.2f}",
                f"{_pc_curr_sym} {_pc_labour:,.2f}",
                f"{_pc_curr_sym} {_pc_irrig:,.2f}",
                f"{_pc_curr_sym} {_pc_other:,.2f}",
                f"{_pc_curr_sym} {_pc_total_expenses:,.2f}",
                f"{_pc_curr_sym} {_pc_profit:,.2f}",
                f"{_pc_margin:.2f}%",
            ],
        })
        st.dataframe(_summary_df, use_container_width=True, hide_index=True)

    # ----------------------------------------------------------------
    # STEP 12  Disclaimers
    # ----------------------------------------------------------------

    st.markdown(
        """
        <div class="agri-callout">
            ℹ️ <strong>Important Disclaimers:</strong>
            All values shown here are <strong>estimates only</strong>. They are based
            on the predicted yield from the ML model and the figures you entered.
            Actual harvest, revenue and profit depend on many real-world factors
            including weather, pest pressure, market price fluctuations, post-harvest
            losses, transportation and storage costs. This calculator is a planning
            aid, not a financial guarantee.
            &nbsp;&mdash;&nbsp;
            <strong>Unit note:</strong> The model predicts yield in hg/ha
            (hectograms per hectare). 1 hg = 0.1 kg, so 1 hg/ha over 1 hectare
            produces 0.1 kg of crop.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
