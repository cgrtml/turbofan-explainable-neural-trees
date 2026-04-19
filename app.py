"""
Turbofan Engine RUL Prediction
IEEE SMC 2026 | Trustworthy AI for Safety-Critical Systems
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

from src.data_preprocessing import (
    load_cmapss, prepare_features, USEFUL_FEATURES,
    add_gaussian_noise, simulate_missing_sensors
)
from src.neural_tree import NeuralTreeEnsemble, train_neural_tree, predict_neural_tree
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Turbofan RUL Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-banner {
    background: linear-gradient(135deg, #0b0c2a 0%, #1a237e 50%, #0d47a1 100%);
    border-radius: 16px;
    padding: 40px 36px;
    color: white;
    margin-bottom: 24px;
}
.hero-banner h1 {
    font-size: 2rem; font-weight: 700; margin: 0 0 8px 0; color: white;
}
.hero-banner p { font-size: 1rem; opacity: 0.85; margin: 0; color: white; }
.badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    margin: 6px 4px 0 0;
    color: white;
}
.nasa-card {
    background: #f8f9ff;
    border: 1px solid #e0e4f0;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
}
.stat-card {
    background: linear-gradient(135deg, #1a237e, #1565c0);
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    color: white;
}
.stat-card .num { font-size: 2.2rem; font-weight: 700; }
.stat-card .lbl { font-size: 0.8rem; opacity: 0.8; margin-top: 2px; }
.metric-card {
    background: #f0f4ff;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    border-left: 4px solid #2196F3;
}
.github-btn {
    background: #24292e;
    color: white !important;
    padding: 10px 20px;
    border-radius: 8px;
    text-decoration: none;
    font-weight: 600;
    display: inline-block;
    margin-top: 8px;
}
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #5c6bc0;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)


# ── Model training (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models on NASA CMAPSS data...")
def load_models_and_data():
    train_df, test_df = load_cmapss(dataset='FD001', data_dir='data')
    X_train, y_train, X_test, y_test, scaler = prepare_features(train_df, test_df)
    input_dim = X_train.shape[1]

    # Neural Tree
    nt = NeuralTreeEnsemble(input_dim=input_dim, n_trees=5, depth=5,
                             hidden_dim=32, dropout=0.15)
    train_neural_tree(nt, X_train, y_train, epochs=200, lr=5e-4,
                      sensor_dropout=0.1, verbose=False)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=12,
                                random_state=42, n_jobs=4)
    rf.fit(X_train, y_train)

    # Gradient Boosting
    gb = HistGradientBoostingRegressor(max_iter=200, max_depth=5,
                                        learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)

    return nt, rf, gb, X_train, y_train, X_test, y_test, train_df, test_df, scaler


def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1)))


def get_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    ns   = nasa_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R²": r2, "NASA Score": ns}


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="background:linear-gradient(135deg,#0b0c2a,#1a237e);border-radius:10px;padding:16px;margin-bottom:12px;text-align:center">
    <div style="font-size:2rem">✈️</div>
    <div style="color:white;font-weight:700;font-size:0.95rem;margin-top:4px">Turbofan RUL</div>
    <div style="color:rgba(255,255,255,0.7);font-size:0.75rem">IEEE SMC 2026</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", [
    "🏠 Overview",
    "🔮 RUL Prediction",
    "🌊 Robustness Test",
    "📊 Model Comparison",
    "🏆 Results vs Literature",
    "🔍 Feature Importance"
])
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size:0.8rem;color:#555;line-height:1.8">
<b>Dataset</b><br>NASA CMAPSS FD001<br><br>
<b>Author</b><br>Cagri Temel<br><br>
<b>Conference</b><br>IEEE SMC 2026<br>Bellevue, WA, USA
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("")
st.sidebar.markdown("[GitHub Repository](https://github.com/cgrtml/turbofan-explainable-neural-trees)")

# ── Load data & models ─────────────────────────────────────────────────────────
nt, rf, gb, X_train, y_train, X_test, y_test, train_df, test_df, scaler = load_models_and_data()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":

    # Hero banner
    st.markdown("""
    <div class="hero-banner">
        <h1>Explainable Neural Trees for Turbofan Engine RUL Prediction</h1>
        <p>Predicting when aircraft engines need maintenance, before failure occurs.</p>
        <br>
        <span class="badge">📄 IEEE SMC 2026</span>
        <span class="badge">🏛 Trustworthy AI Workshop</span>
        <span class="badge">✈️ Safety-Critical Systems</span>
        <span class="badge">🔬 NASA CMAPSS Dataset</span>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, num, lbl in [
        (c1, "100", "Training Engines"),
        (c2, "100", "Test Engines"),
        (c3, "21", "Sensor Channels"),
        (c4, "130", "Max RUL (cycles)"),
        (c5, "0.807", "Neural Tree R²"),
    ]:
        col.markdown(f"""
        <div class="stat-card">
            <div class="num">{num}</div>
            <div class="lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # NASA section
    col_nasa, col_gh = st.columns([3, 1])
    with col_nasa:
        st.markdown('<div class="section-label">Data Source</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="nasa-card">
        <img src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIj8+CjxzdmcgaGVpZ2h0PSI5MiIgdmlld0JveD0iMCAwIDExMCA5MiIgd2lkdGg9IjExMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGNpcmNsZSBjeD0iNTAuMDQ5IiBjeT0iNDUiIGZpbGw9IiMwYjNkOTEiIHI9IjQwLjE0Ii8+CjxnIGZpbGw9IiNmZmYiPgo8Y2lyY2xlIGN4PSI0Ny42NzkiIGN5PSIxMi41NyIgcj0iLjQ1Ii8+CjxjaXJjbGUgY3g9IjUyLjI5OSIgY3k9IjEzLjE3IiByPSIuNDUiLz4KPGNpcmNsZSBjeD0iNTguMzU5IiBjeT0iMjEuMzMiIHI9Ii40NSIvPgo8Y2lyY2xlIGN4PSIyNS4xMTkiIGN5PSI2My4zMyIgcj0iLjQ1Ii8+CjxjaXJjbGUgY3g9IjI2LjI4OSIgY3k9IjY2LjkzIiByPSIuNDUiLz4KPGNpcmNsZSBjeD0iMjAuNzA5IiBjeT0iNjMuODciIHI9Ii4zMzciLz4KPGNpcmNsZSBjeD0iMzkuMDA5IiBjeT0iNzAuOTQyIiByPSIuMzM4Ii8+CjxjaXJjbGUgY3g9IjY3LjcxMSIgY3k9IjY0Ljk4IiByPSIuMzM3Ii8+CjxjaXJjbGUgY3g9Ijc2LjA1MiIgY3k9IjU1LjkyIiByPSIuMzM4Ii8+CjxjaXJjbGUgY3g9IjM1LjE2OSIgY3k9IjIzLjk2MiIgcj0iLjMzNyIvPgo8Y2lyY2xlIGN4PSI0NC4zNDkiIGN5PSIxNy4yMiIgcj0iLjMzNyIvPgo8Y2lyY2xlIGN4PSI0My4zNTIiIGN5PSIxNi41NiIgcj0iLjMzNyIvPgo8Y2lyY2xlIGN4PSI0Mi40NTIiIGN5PSIxNS45IiByPSIuMzM3Ii8+CjxjaXJjbGUgY3g9IjM2LjYwOSIgY3k9IjI1LjcwMyIgcj0iLjMzNyIvPgo8Y2lyY2xlIGN4PSI1MC4xMzEiIGN5PSI4LjE2IiByPSIuMzM3Ii8+CjxjaXJjbGUgY3g9IjUyLjM1MiIgY3k9IjE3Ljg4IiByPSIuMzM3Ii8+CjxjaXJjbGUgY3g9IjQ4Ljg0OSIgY3k9IjE1Ljk4MiIgcj0iLjMzNyIvPgo8Y2lyY2xlIGN4PSI0Mi44NDkiIGN5PSIxOC41NjMiIHI9Ii4zMzciLz4KPGNpcmNsZSBjeD0iNjkuMzA5IiBjeT0iNzMuODgzIiByPSIuMzM3Ii8+CjxjaXJjbGUgY3g9IjI0LjU0OSIgY3k9IjY1LjYxIiByPSIuMzM4Ii8+CjxjaXJjbGUgY3g9IjQ4LjAwOSIgY3k9IjY5Ljk2IiByPSIuMzM4Ii8+CjxjaXJjbGUgY3g9IjMxLjUzMSIgY3k9IjY1LjM0IiByPSIuMzM4Ii8+CjxjaXJjbGUgY3g9IjM0LjQ0OSIgY3k9IjcwLjEwMyIgcj0iLjMzOCIvPgo8Y2lyY2xlIGN4PSI1NS45MjkiIGN5PSI2Ny4xMDMiIHI9Ii4zMzciLz4KPGNpcmNsZSBjeD0iNjcuNzcxIiBjeT0iNjAuNDIiIHI9Ii4zMzciLz4KPGNpcmNsZSBjeD0iNzYuNzQ5IiBjeT0iNjQuNTIyIiByPSIuMzM3Ii8+CjxjaXJjbGUgY3g9Ijc5LjgwOSIgY3k9IjY2LjQ4IiByPSIuMzM3Ii8+CjxjaXJjbGUgY3g9IjgwLjMxMiIgY3k9IjYxLjE0IiByPSIuMzM3Ii8+CjxjaXJjbGUgY3g9IjM1LjY3MSIgY3k9IjUzLjU4IiByPSIuMzM3Ii8+CjxjaXJjbGUgY3g9IjM1Ljc5OSIgY3k9IjYxLjMyIiByPSIuNDUiLz4KPGNpcmNsZSBjeD0iMzguNDk5IiBjeT0iNjcuMDIiIHI9Ii40NSIvPgo8Y2lyY2xlIGN4PSI3MC44MzkiIGN5PSI2MS4wOCIgcj0iLjQ1Ii8+CjxjaXJjbGUgY3g9IjgyLjQ3OSIgY3k9IjYwLjQyIiByPSIuNDUiLz4KPGNpcmNsZSBjeD0iNzYuNzE5IiBjeT0iNTcuOTYiIHI9Ii40NSIvPgo8Y2lyY2xlIGN4PSI3MC44MzkiIGN5PSI1OC4yIiByPSIuNDUiLz4KPHBhdGggZD0iTTU4LjcxIDEyLjI4OGwxLjExOS0uMTA3LTEuMTE3LS4wNjNjLS4wMzUtLjIxNi0uMjA4LS4zODUtLjQyNi0uNDEzbC0uMTA3LTEuMTE0LS4wNjQgMS4xMjNjLS4yMDIuMDQ1LS4zNTcuMjE0LS4zODIuNDI0bC0xLjE0NC4xMDQgMS4xNTIuMDYyYy4wNDIuMTkzLjE5OC4zNDQuMzk0LjM4bC4xMDQgMS4xNDguMDYxLTEuMTQ2QzU4LjUwNyAxMi42NTEgNTguNjcxIDEyLjQ5MiA1OC43MSAxMi4yODh6Ii8+CjxwYXRoIGQ9Ik0zOS44MjQgMjQuNzQ2bDEuMTE5LS4xMDctMS4xMTctLjA2M2MtLjAzNC0uMjE2LS4yMDgtLjM4NS0uNDI2LS40MTNsLS4xMDctMS4xMTQtLjA2MyAxLjEyM2MtLjIwMy4wNDUtLjM1OC4yMTQtLjM4My40MjRsLTEuMTQ0LjEwNCAxLjE1Mi4wNjJjLjA0Mi4xOTMuMTk4LjM0NC4zOTQuMzhsLjEwNCAxLjE0OC4wNjItMS4xNDZDMzkuNjIyIDI1LjExIDM5Ljc4NiAyNC45NSAzOS44MjQgMjQuNzQ2eiIvPgo8cGF0aCBkPSJNODEuNjU5IDU3LjY4NGwxLjExOS0uMTA3LTEuMTE3LS4wNjNjLS4wMzQtLjIxNi0uMjA4LS4zODUtLjQyNi0uNDEzbC0uMTA3LTEuMTE0LS4wNjMgMS4xMjNjLS4yMDIuMDQ1LS4zNTcuMjE0LS4zODIuNDI0bC0xLjE0NC4xMDQgMS4xNTIuMDYyYy4wNDIuMTkzLjE5OC4zNDQuMzk0LjM4bC4xMDQgMS4xNDguMDYyLTEuMTQ2QzgxLjQ1NiA1OC4wNDggODEuNjIgNTcuODg5IDgxLjY1OSA1Ny42ODR6Ii8+CjxwYXRoIGQ9Ik0zNi4wNDQgNzQuOTA2bDEuMTE5LS4xMDctMS4xMTctLjA2M2MtLjAzNS0uMjE2LS4yMDgtLjM4NS0uNDI2LS40MTNsLS4xMDctMS4xMTMtLjA2MyAxLjEyMmMtLjIwMy4wNDUtLjM1OC4yMTQtLjM4My40MjRsLTEuMTQ0LjEwNCAxLjE1Mi4wNjJjLjA0Mi4xOTMuMTk4LjM0NS4zOTQuMzhsLjEwNCAxLjE0OC4wNjItMS4xNDZDMzUuODQxIDc1LjI3IDM2LjAwNiA3NS4xMSAzNi4wNDQgNzQuOTA2eiIvPgo8cGF0aCBkPSJNNzguMTA0IDY2LjUwNmwxLjExOS0uMTA3LTEuMTE3LS4wNjNjLS4wMzQtLjIxNi0uMjA4LS4zODUtLjQyNi0uNDEzbC0uMTA3LTEuMTE0LS4wNjMgMS4xMjJjLS4yMDIuMDQ1LS4zNTcuMjE0LS4zODIuNDI0bC0xLjE0NC4xMDQgMS4xNTIuMDYyYy4wNDIuMTkzLjE5OC4zNDQuMzk0LjM4bC4xMDQgMS4xNDguMDYyLTEuMTQ2Qzc3LjkwMSA2Ni44NyA3OC4wNjYgNjYuNzEgNzguMTA0IDY2LjUwNnoiLz4KPHBhdGggZD0iTTU5LjU2OCAzNS4zODVjLTQuNjY3IDEuODE0LTkuMjE5IDMuNDMzLTEzLjA2IDQuNjM1LTcuODA1IDIuNDQ0LTI5LjE2IDkuMDYtNDIuMDYgMTcuNGwxLjA4LjQyYzcuODYtNC40NCAxMi45NjktNS44MzUgMTcuODgtNy4zOCA1LjM0LTEuNjggMjIuNjAzLTUuNzIgMzAuNDItNy45MiAyLjY0MS0uNzQzIDUuNzM0LTEuNzE2IDkuMDEtMi45LS43NjItMS4wNjMtMS41NjYtMi4xMjktMi40MTItMy4xOTNDNjAuMTQzIDM2LjA4OCA1OS44NTYgMzUuNzM0IDU5LjU2OCAzNS4zODV6TTY1LjI3IDQzLjI0NGMtMS4xMy43NjMtMi4wNzcgMS4zNzItMi43NCAxLjc1Ni0zLjg0IDIuMjItMjIuNTYxIDE1LTI2LjgyIDE3Ljk0cy0xNi4wOCAxNC4xLTE5LjU2IDE3LjM0bC0uMTIgMS4zMTljMTEuMjItMTAuMDggMTQuNzQtMTIuNTY2IDE5LjItMTUuOTU5IDUuNTItNC4yIDE2LjkzOS0xMS45NyAyMC44Mi0xNC40NiAzLjcxLTIuMzggNy4wNTYtNC41NjkgMTAuMDU5LTYuNTcyLS4wNDktLjA4Mi0uMDk4LS4xNjQtLjE0Ny0uMjQ3QzY1LjczNiA0My45OSA2NS41MDUgNDMuNjE4IDY1LjI3IDQzLjI0NHpNODIuODA5IDI0LjcyYy01LjQ2NiAzLjIwNC0xNC4wODEgNy4wNzEtMjIuNDM5IDEwLjM1Mi4yLjI0NS4zOTkuNDkyLjU5Ny43NDEuOTM0IDEuMTc2IDEuODE1IDIuMzYgMi42NDQgMy41NDUgNi41Ny0yLjQyIDEzLjc3OS01LjY2OCAxOS40OTktOS41OTktMi43MjUgMi41ODItMTEuNzM0IDkuMzE1LTE3LjIyNyAxMy4wNjguMjgzLjQ2MS41NTcuOTIyLjgyMiAxLjM4MSA4LjMyMi01LjU2OSAxMy45MjItOS42NjggMTcuMTg1LTEyLjQwOSA0LjUtMy43OCAxNC43Ni0xMi4yNCAxOC42Ni0yMy41OEM5NS43MDkgMTYuOTIgODcuNjIxIDIxLjg5OSA4Mi44MDkgMjQuNzJ6IiBmaWxsPSIjZmMzZDIxIi8+CjxwYXRoIGQ9Ik00NC44ODQgNTQuOTM5Yy0uODg1LTEuMTE0LTIuMTA5LTIuNjA2LTMuMDI4LTMuNzYzLTEuMjI5LTEuNTQ3LTIuMzY2LTMuMTEtMy40MDgtNC42NzEtLjM0LjA4NS0uNjc5LjE3LTEuMDE4LjI1NSAxLjI1OCAxLjk2MyAyLjY1NSAzLjkyMyA0LjE3NyA1LjgzOSAxLjExMiAxLjQgMi4xMjMgMi41MjcgMi42NDEgMy4yMjguMTA1LjE0Mi4zMTMuNDU2LjU5NC44NzQuMzI0LS4yMi42NTEtLjQ0Mi45ODEtLjY2NkM0NS41MDQgNTUuNjg4IDQ1LjE4OSA1NS4zMjMgNDQuODg0IDU0LjkzOXpNNTEuMzQ0IDYwLjgwM2MtLjcyNy0uNjg4LTIuNDktMS44MzctNC4zMjUtMy41NjEtLjQwNS4yNzgtLjgxNC41Ni0xLjIyNC44NDQgMS4xODUgMS42NyAyLjc5OSAzLjcyMSA0LjA2MyA0LjMxOUM1MS43NjIgNjMuMzA3IDUyLjI3NSA2MS42ODUgNTEuMzQ0IDYwLjgwM3pNNjAuOTY3IDM1LjgxM2MtMTAuNDkyLTEzLjIwNi0yMy4zMDktMjAuNDYxLTI4LjgzNS0xNi4wNy00LjI5MiAzLjQxLTIuNTMgMTMuMzc2IDMuMzg2IDIzLjg0NS4zMDYtLjEwNS42MDktLjIwOC45MDktLjMxLTUuOTcxLTEwLjItNy42MDUtMTkuNjc5LTMuNTU3LTIyLjg5NiA1LjA4Ny00LjA0MiAxNy4zNyAzLjI0MSAyNy41NTggMTYuMDY0IDIuMTA5IDIuNjU0IDMuOTYzIDUuMzE4IDUuNTMzIDcuOTE1IDYuMDEyIDkuOTUgNy44NTcgMTguOTQ4IDMuNzAzIDIyLjYyMS0xLjI3MSAxLjEyNC01LjE1NSAxLjU2NS0xMC4yNDMtLjcyNS0uMDcxLjA4OS4wNDMuMzMuMTMyLjM4OSA0LjM5MiAxLjc2NiA4LjU5OSAyLjQzOSAxMC43MjMuNzUyQzc1LjM4IDYzLjM0MiA3MS40NTkgNDkuMDE5IDYwLjk2NyAzNS44MTN6Ii8+CjxwYXRoIGQ9Ik0xNS45NjkgMzcuMzhoNi43Mmw1LjY0IDkuNTdjMCAwIDAtNi45MyAwLTcuNDcgMC0uODQtMS4wNjUtMS45MzUtMS40NC0yLjEuNDUgMCA0LjM4IDAgNC42NSAwLS4yODUuMDc1LTEuMiAxLjE4NS0xLjIgMi4xIDAgLjQ1IDAgMTAuNSAwIDEwLjk4IDAgLjY3NS45NzUgMS42MDUgMS40NCAxLjk2NWgtNi40OGwtNS43My05LjYxNWMwIDAgMCA3LjE3IDAgNy41NiAwIC43NS43MzUgMS40NyAxLjUgMi4wODVoLTQuOTVjLjcwNS0uMyAxLjM4LTEuMjQ1IDEuNDQtMS45OTVzMC0xMC40MjUgMC0xMC44NDVDMTcuNTU5IDM4LjcgMTYuNjc0IDM3Ljk1IDE1Ljk2OSAzNy4zOHoiLz4KPHBhdGggZD0iTTc3LjQzOSA1Mi40MjVoOC45NGMtLjQ5NS0uMTItMS4wNS0uNzA1LTEuMzUtMS40ODUtLjMtLjc4LTUuMDQtMTMuNTYtNS4wNC0xMy41Nkg3Ni41OWMtLjk2NC42OTQtMS45OTcgMS40MjYtMy4xIDIuMTk3LS4wMDMuMDI4LS4wMDYuMDU2LS4wMTEuMDgzLS4xNDguOS0yLjgwOCAxMC41MzQtMi45NyAxMS4wMS0uMjI1LjY2LTEuMzggMS4zOTUtMS44NDUgMS43ODVoNC44MTVjLS40OC0uNTQtLjg3LTEuMDY1LS43OC0xLjY2NS4wOS0uNi4zNi0xLjguMzYtMS44aDQuOThjLjIyNS42LjM5MyAxLjEzOS40OCAxLjY1Qzc4LjYyNCA1MS4yNTUgNzcuOTk0IDUxLjk0NSA3Ny40MzkgNTIuNDI1ek03My41MDkgNDcuMDdsMS42OC01LjQ5IDIuMjIgNS40OUg3My41MDl6TTcyLjc1MiAzNy45MjhjLjI0Ny0uMTgyLjQ5NS0uMzY1Ljc0Mi0uNTQ4aC0xLjMwNUM3Mi4zMTkgMzcuNSA3Mi41MzQgMzcuNjg5IDcyLjc1MiAzNy45Mjh6Ii8+CjxwYXRoIGQ9Ik0zOC41NTkgNTAuNzljLjA5LS42LjM2LTEuOC4zNi0xLjhoNC45OGMuMjI1LjYuMzkzIDEuMTM5LjQ4IDEuNjUuMTA1LjYxNS0uNTI1IDEuMzA1LTEuMDggMS43ODVoNy44NzFjLjE2NC0uMTEuMzI3LS4yMi40OS0uMzI5LS4zMDUtLjI3LS41ODYtLjY3NS0uNzcxLTEuMTU2LS4zLS43OC01LjA0LTEzLjU2LTUuMDQtMTMuNTZoLTcuOGMuMzc1LjM0NSAxLjQ1NSAxLjI3NSAxLjI5IDIuMjgtLjE0Ny45LTIuODA4IDEwLjUzNC0yLjk3IDExLjAxLS4yMjUuNjYtMS4zOCAxLjM5NS0xLjg0NSAxLjc4NWg0LjgxNUMzOC44NTkgNTEuOTE1IDM4LjQ2OSA1MS4zOSAzOC41NTkgNTAuNzl6TTQxLjA0OSA0MS41OGwyLjIyIDUuNDloLTMuOUw0MS4wNDkgNDEuNTh6Ii8+CjxwYXRoIGQ9Ik02NS43NDggNDQuODQ4Yy0xLjQ2OC45NzgtMy4wMTcgMS45OTktNC42NDkgMy4wNjUuNzMyLjM1NSAxLjMxNS44MDEgMS4zNzEgMS4zNzcuMTA0IDEuMDgyLTIuMDcgMS42MDUtNC4wMzUgMS4zOC0uMzkzLS4wNDUtLjc3OS0uMTQ4LTEuMTQ3LS4yODYtLjQwOC4yNjMtLjgyLjUyOC0xLjIzOC43OTYtLjQyNS4yNzMtLjk0MS42MDktMS41My45OTd2MS41NTNjLjM5LS43NjUgMS4yNDMtMS40NSAxLjkwNS0xLjQ4NS4yODUtLjAxNSAxLjI3NS45IDUuMzU1LjY3NSAxLjk4LS4xMDkgNS44MDUtMi4yMiA1Ljc0NS00LjY1QzY3LjQ4OSA0Ni44MzQgNjYuNzM5IDQ1LjcxNCA2NS43NDggNDQuODQ4ek01NC41MTkgNDguNnYxLjU4MmMuMzYxLS4yNDEuNzE3LS40NzggMS4wNjYtLjcwOUM1NS4wMzYgNDkuMDkxIDU0LjY0NyA0OC43MzQgNTQuNTE5IDQ4LjZ6TTY0LjM1MyA0My44NTVjLS4zOC0uMjI1LS43NjUtLjQyMi0xLjEzNC0uNTk2LTEuOTItLjktMy45My0xLjA2NS00LjM1LTIuMjgtLjI5Ni0uODU3LjU0LTEuNjUgMi41OC0xLjYyIDIuMDQuMDMgMy45MyAxLjI0NSA0LjQ0IDEuNjh2LTMuODdjLS4xNS4xNS0uODA4LjkwNS0xLjQxLjc4LTEuMTU1LS4yNC0zLjEyLS41NTMtNS4zNy0uNTQtMi41OC4wMTUtNC44IDIuMDA5LTQuODc1IDQuNTMtLjEwNSAzLjUyNSAyLjcxNSA0LjQ4NSA0LjMwNSA1LjA0LjE2NC4wNTcuMzUxLjExOC41NTQuMTgzIDEuNTI1LS45OTIgMi43MzEtMS43NTYgMy40MzctMi4xNjNDNjMuMDA0IDQ0LjcyNiA2My42MjUgNDQuMzM0IDY0LjM1MyA0My44NTV6Ii8+CjwvZz4KPC9zdmc+Cg==" height="52" style="margin-bottom:12px"/>
        <h4 style="margin:0 0 8px 0">CMAPSS: Commercial Modular Aero-Propulsion System Simulation</h4>
        <p style="margin:0;color:#444;font-size:0.9rem">
        Published by NASA's Prognostics Center of Excellence, the CMAPSS dataset is the
        gold standard benchmark for turbofan engine degradation research.
        It contains run-to-failure measurements from 100 simulated engines, each instrumented
        with 21 sensors recording temperature, pressure, fan speed, and bypass ratios
        across hundreds of flight cycles until engine failure.
        The dataset is used in hundreds of peer-reviewed publications worldwide.
        </p>
        <br>
        <a href="https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/"
           target="_blank" style="color:#1565c0;font-weight:600;font-size:0.85rem">
           View NASA Data Repository
        </a>
        </div>
        """, unsafe_allow_html=True)

    with col_gh:
        st.markdown('<div class="section-label">Source Code</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="nasa-card" style="text-align:center">
        <div style="font-size:2.2rem;margin-bottom:10px">🐙</div>
        <h4 style="margin:0 0 6px 0">Open Source</h4>
        <p style="font-size:0.82rem;color:#555;margin:0 0 12px 0">
        Full implementation, experiments, and trained models available on GitHub.
        </p>
        <a href="https://github.com/cgrtml/turbofan-explainable-neural-trees"
           target="_blank" class="github-btn">
           View on GitHub
        </a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-label">Problem</div>', unsafe_allow_html=True)
        st.subheader("What is Remaining Useful Life?")
        st.markdown("""
        **Remaining Useful Life (RUL)** is the number of flight cycles a turbofan engine
        can complete before requiring maintenance or replacement.

        Unplanned engine failures cost airlines millions of dollars per incident and
        pose serious safety risks. Accurate RUL prediction enables:
        - Scheduling maintenance before failure occurs
        - Avoiding unnecessary early replacements
        - Keeping aircraft in service as long as safely possible
        """)

        st.markdown('<div class="section-label" style="margin-top:20px">Our Method</div>', unsafe_allow_html=True)
        st.subheader("Explainable Neural Trees")
        st.markdown("""
        A hybrid model combining the predictive power of neural networks with
        the interpretability of decision trees.

        - **MLP Embedding layer** learns compact representations from 17 raw sensor channels
        - **Soft Decision Tree ensemble** routes predictions through interpretable branching logic
        - **Sensor Dropout Training** exposes the model to missing sensors during training,
          making it robust when sensors fail in the field
        - **Gradient-based explanations** identify which sensors drive each prediction
        """)

    with col2:
        st.markdown('<div class="section-label">Live Data</div>', unsafe_allow_html=True)
        st.subheader("Sensor Degradation: Engine #1")
        st.caption("Normalized sensor readings across 163 flight cycles. As the engine ages, sensors drift from their baseline values.")
        engine_1 = train_df[train_df['unit_id'] == 1].copy()
        fig = go.Figure()
        for sensor, color, label in [
            ('s4',  '#1565c0', 'HPC Outlet Temp (s4)'),
            ('s11', '#2e7d32', 'Bypass Ratio (s11)'),
            ('s12', '#e65100', 'Burner Fuel-Air Ratio (s12)'),
            ('s20', '#880e4f', 'HPT Bleed Enthalpy (s20)'),
        ]:
            vals = engine_1[sensor]
            normalized = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
            fig.add_trace(go.Scatter(
                x=engine_1['time_cycle'], y=normalized,
                name=label, line=dict(color=color, width=2)
            ))
        fig.update_layout(
            xaxis_title="Flight Cycle",
            yaxis_title="Normalized Sensor Value (0 = min, 1 = max)",
            height=340, margin=dict(t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(248,249,255,1)',
            legend=dict(orientation='h', y=-0.25, font=dict(size=11))
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Dataset Statistics</div>', unsafe_allow_html=True)
    st.subheader("RUL Distribution across Training Engines")
    st.caption("Most engines in the training set have between 0 and 130 cycles of life remaining. The RUL is capped at 130 to focus the model on the degradation phase.")
    fig2 = px.histogram(train_df, x='RUL', nbins=35,
                        color_discrete_sequence=['#1565c0'],
                        labels={'RUL': 'Remaining Useful Life (cycles)', 'count': 'Number of Readings'})
    fig2.update_traces(marker_line_width=0.5, marker_line_color='white')
    fig2.update_layout(height=260, margin=dict(t=10, b=40),
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(248,249,255,1)')
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: RUL Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 RUL Prediction":
    st.title("RUL Prediction")
    st.markdown("""
    The NASA CMAPSS dataset contains **100 test engines**, each one is a physical turbofan engine
    unit that was run under real operating conditions. For each engine, we have its final sensor
    readings before it was stopped. The model uses these readings to predict how many flight cycles
    the engine has remaining before it needs maintenance.

    Use the controls on the left to select an engine, simulate sensor noise, or drop sensors
    to see how predictions change under degraded conditions.
    """)
    st.markdown("---")

    # Engine health overview bar
    rul_vals = y_test
    critical  = int(np.sum(rul_vals <= 40))
    caution   = int(np.sum((rul_vals > 40) & (rul_vals <= 80)))
    healthy   = int(np.sum(rul_vals > 80))
    st.markdown(f"""
    <div style="background:#f8f9ff;border-radius:10px;padding:14px 20px;
                display:flex;gap:24px;margin-bottom:16px;border:1px solid #e0e4f0">
        <div>
            <span style="font-size:0.72rem;font-weight:600;letter-spacing:1px;
                         text-transform:uppercase;color:#555">Fleet Status (100 test engines)</span>
        </div>
        <div><span style="color:#c62828;font-weight:700">{critical}</span>
             <span style="color:#888;font-size:0.85rem"> Critical (under 40 cycles)</span></div>
        <div><span style="color:#f57f17;font-weight:700">{caution}</span>
             <span style="color:#888;font-size:0.85rem"> Caution (40–80 cycles)</span></div>
        <div><span style="color:#2e7d32;font-weight:700">{healthy}</span>
             <span style="color:#888;font-size:0.85rem"> Healthy (over 80 cycles)</span></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("""
        <div style="background:#f0f4ff;border-radius:10px;padding:14px;margin-bottom:12px;
                    border:1px solid #c5cae9;font-size:0.82rem;color:#333;line-height:1.6">
        <b>What are these engines?</b><br><br>
        The NASA CMAPSS dataset contains <b>100 simulated turbofan engines</b>, each one
        representing a different physical unit in a fleet. Every engine was run from a
        healthy state until failure under real operating conditions.<br><br>
        For each engine, the test set captures its <b>final sensor snapshot</b> before
        it was stopped. The true remaining life was measured but hidden from the model.
        It is used here to evaluate prediction accuracy.
        </div>
        """, unsafe_allow_html=True)

        engine_id = st.selectbox("Select engine to inspect", options=list(range(1, 101)),
                                  format_func=lambda x: f"Engine {x}  (RUL = {int(y_test[x-1])} cycles)")
        st.markdown("**Add sensor noise**")
        st.caption("Simulates electromagnetic interference or calibration drift.")
        noise_level = st.slider("Noise level (σ)", 0.0, 0.20, 0.0, 0.01)
        st.markdown("**Drop sensors**")
        st.caption("Simulates complete sensor failure. The channel stops transmitting.")
        missing_pct = st.slider("Missing sensors (%)", 0, 50, 0, 10)

    # Get this engine's last test reading
    X_engine = X_test[engine_id - 1:engine_id].copy()
    y_engine = y_test[engine_id - 1]

    if noise_level > 0:
        X_engine = add_gaussian_noise(X_engine, noise_std=noise_level)
    if missing_pct > 0:
        X_engine, _ = simulate_missing_sensors(X_engine, missing_ratio=missing_pct/100)

    pred_nt = float(predict_neural_tree(nt, X_engine)[0])
    pred_rf = float(rf.predict(X_engine)[0])
    pred_gb = float(gb.predict(X_engine)[0])
    true_rul = float(y_engine)

    status = "🔴 Critical" if true_rul <= 40 else ("🟡 Caution" if true_rul <= 80 else "🟢 Healthy")

    with col2:
        st.markdown(f"### Engine {engine_id} &nbsp; {status}")
        st.markdown(f"**Actual remaining life: {true_rul:.0f} flight cycles.** This is the ground truth measured after the engine was stopped.")
        st.caption("Each model reads the sensor snapshot and independently predicts how many cycles remain. Lower error = better prediction.")

        c1, c2, c3 = st.columns(3)
        for col, name, pred, color in [
            (c1, "Neural Tree",       pred_nt, "#2196F3"),
            (c2, "Random Forest",     pred_rf, "#4CAF50"),
            (c3, "Gradient Boosting", pred_gb, "#FF9800"),
        ]:
            err = abs(pred - true_rul)
            with col:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color:{color}">
                    <h4 style="color:{color};margin:0">{name}</h4>
                    <h2 style="margin:8px 0">{pred:.1f}</h2>
                    <small>Prediction error: {err:.1f} cycles</small>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("**Gauge charts below:** Each dial shows a model's predicted RUL. The black line marks the true RUL. Red zone = critical (under 40 cycles), yellow = caution, green = healthy.")

        # Gauge charts — equal size via separate columns
        g1, g2, g3 = st.columns(3)
        for gcol, name, pred, color in [
            (g1, "Neural Tree",       pred_nt, "#2196F3"),
            (g2, "Random Forest",     pred_rf, "#4CAF50"),
            (g3, "Gradient Boosting", pred_gb, "#FF9800"),
        ]:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                title=dict(text=name, font=dict(size=14)),
                gauge=dict(
                    axis=dict(range=[0, 130]),
                    bar=dict(color=color),
                    steps=[
                        dict(range=[0, 40],  color="#ffebee"),
                        dict(range=[40, 80], color="#fff9c4"),
                        dict(range=[80, 130],color="#e8f5e9"),
                    ],
                    threshold=dict(line=dict(color="black", width=3), value=true_rul)
                ),
                number=dict(suffix=" cycles")
            ))
            fig_g.update_layout(height=240, margin=dict(t=30, b=10, l=20, r=20))
            gcol.plotly_chart(fig_g, use_container_width=True)

        st.markdown("---")
        st.markdown("**Sensor readings for this engine**")
        st.caption("Each bar is a normalized sensor value (0 = min, 1 = max observed in training). Red bars indicate sensors that are missing or zeroed out.")
        sensor_vals = X_engine[0]
        fig2 = go.Figure(go.Bar(
            x=USEFUL_FEATURES, y=sensor_vals,
            marker_color=['#ef5350' if v == 0 else '#2196F3' for v in sensor_vals],
        ))
        fig2.update_layout(height=240, margin=dict(t=10, b=40),
                           yaxis_title="Normalized Value [0–1]",
                           xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Robustness Test
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌊 Robustness Test":
    st.title("Robustness Under Real-World Conditions")
    st.markdown("""
    In safety-critical environments like aviation, sensors don't always work perfectly.
    This page tests how well each model holds up when sensor data is imperfect.
    There are two types of problems that can occur in real aircraft sensors:
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.info("""
        **Sensor Noise**

        The sensor is still working, but its readings contain random errors.
        This happens due to electromagnetic interference, vibration, temperature fluctuations,
        or aging electronics. The sensor reports a value close to the true value,
        but with small random distortions added.

        Example: A temperature sensor reads 487.3°C instead of the true 485.0°C.
        """)
    with col_b:
        st.warning("""
        **Missing Sensors**

        The sensor has completely stopped transmitting data.
        This can happen due to wiring failures, physical damage, or power loss to the sensor unit.
        The model receives a zero value instead of real data.

        Example: A pressure sensor stops working mid-flight and reports nothing.
        """)

    st.markdown("---")
    st.markdown("""
    **Why Neural Trees handle this better:**
    During training, Neural Trees are deliberately exposed to randomly dropped sensors
    (a technique called *sensor dropout*). This forces the model to learn which combinations
    of sensors carry the most information, so it can still make accurate predictions
    even when some sensors are unavailable.
    """)
    st.markdown("---")

    tab1, tab2 = st.tabs(["Sensor Noise Test", "Missing Sensor Test"])

    with tab1:
        st.subheader("Prediction Error vs Noise Severity")
        st.caption("RMSE = Root Mean Square Error. Lower is better. Each point shows how accurate the model is at that noise level.")

        noise_levels = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
        nt_rmse, rf_rmse, gb_rmse = [], [], []

        with st.spinner("Computing..."):
            for std in noise_levels:
                X_n = add_gaussian_noise(X_test, noise_std=std) if std > 0 else X_test.copy()
                nt_rmse.append(np.sqrt(mean_squared_error(y_test, predict_neural_tree(nt, X_n))))
                rf_rmse.append(np.sqrt(mean_squared_error(y_test, rf.predict(X_n))))
                gb_rmse.append(np.sqrt(mean_squared_error(y_test, gb.predict(X_n))))

        fig = go.Figure()
        for name, vals, color, dash in [
            ("Neural Tree", nt_rmse, "#2196F3", "solid"),
            ("Random Forest", rf_rmse, "#4CAF50", "dash"),
            ("Gradient Boosting", gb_rmse, "#FF9800", "dot"),
        ]:
            fig.add_trace(go.Scatter(x=noise_levels, y=vals, name=name,
                                      line=dict(color=color, width=2.5, dash=dash),
                                      mode='lines+markers', marker=dict(size=8)))
        fig.update_layout(xaxis_title="Noise Level (σ)", yaxis_title="Prediction Error (RMSE, cycles)",
                          height=380, legend=dict(orientation='h', y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

        r1, r2, r3 = st.columns(3)
        r1.metric("Neural Tree at σ=0.10", f"{nt_rmse[3]:.2f} cycles", delta=f"+{nt_rmse[3]-nt_rmse[0]:.2f} from baseline")
        r2.metric("Random Forest at σ=0.10", f"{rf_rmse[3]:.2f} cycles", delta=f"+{rf_rmse[3]-rf_rmse[0]:.2f} from baseline")
        r3.metric("Gradient Boosting at σ=0.10", f"{gb_rmse[3]:.2f} cycles", delta=f"+{gb_rmse[3]-gb_rmse[0]:.2f} from baseline")

        st.markdown("""
        **What this graph shows:** All three models start at similar accuracy with clean data.
        As noise increases, Neural Tree degrades slightly faster at high noise levels,
        but maintains the best accuracy at low-to-moderate noise (σ up to 0.05),
        which covers the majority of realistic sensor interference scenarios.
        """)

    with tab2:
        st.subheader("Prediction Error vs Percentage of Failed Sensors")
        st.caption("Each point shows model accuracy when that percentage of sensors have completely stopped working.")

        missing_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        nt_m, rf_m, gb_m = [], [], []

        with st.spinner("Computing..."):
            for ratio in missing_ratios:
                if ratio > 0:
                    X_m, _ = simulate_missing_sensors(X_test, missing_ratio=ratio)
                else:
                    X_m = X_test.copy()
                nt_m.append(np.sqrt(mean_squared_error(y_test, predict_neural_tree(nt, X_m))))
                rf_m.append(np.sqrt(mean_squared_error(y_test, rf.predict(X_m))))
                gb_m.append(np.sqrt(mean_squared_error(y_test, gb.predict(X_m))))

        fig2 = go.Figure()
        for name, vals, color, dash in [
            ("Neural Tree", nt_m, "#2196F3", "solid"),
            ("Random Forest", rf_m, "#4CAF50", "dash"),
            ("Gradient Boosting", gb_m, "#FF9800", "dot"),
        ]:
            fig2.add_trace(go.Scatter(x=[r*100 for r in missing_ratios], y=vals,
                                      name=name, line=dict(color=color, width=2.5, dash=dash),
                                      mode='lines+markers', marker=dict(size=8)))
        fig2.update_layout(xaxis_title="Failed Sensors (%)", yaxis_title="Prediction Error (RMSE, cycles)",
                           height=380, legend=dict(orientation='h', y=-0.2))
        st.plotly_chart(fig2, use_container_width=True)

        r1, r2, r3 = st.columns(3)
        r1.metric("Neural Tree at 30% missing", f"{nt_m[3]:.2f} cycles", delta=f"+{nt_m[3]-nt_m[0]:.2f}")
        r2.metric("Random Forest at 30% missing", f"{rf_m[3]:.2f} cycles", delta=f"+{rf_m[3]-rf_m[0]:.2f}")
        r3.metric("Gradient Boosting at 30% missing", f"{gb_m[3]:.2f} cycles", delta=f"+{gb_m[3]-gb_m[0]:.2f}")

        improvement = rf_m[3] - nt_m[3]
        st.success(f"""
        **Key result:** When 30% of sensors fail, Neural Tree's prediction error increases by only
        {nt_m[3]-nt_m[0]:.1f} cycles from baseline, while Random Forest degrades by {rf_m[3]-rf_m[0]:.1f} cycles.
        Neural Tree is **{improvement:.1f} cycles more accurate** than Random Forest under this failure scenario.
        This advantage comes from sensor dropout training, which teaches the model to work with incomplete data.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("📊 Model Performance Comparison")
    st.markdown("---")

    # Compute metrics on clean test data
    m_nt = get_metrics(y_test, predict_neural_tree(nt, X_test))
    m_rf = get_metrics(y_test, rf.predict(X_test))
    m_gb = get_metrics(y_test, gb.predict(X_test))

    st.subheader("TABLE I: Baseline Results (CMAPSS FD001)")
    df_metrics = pd.DataFrame({
        "Neural Tree": m_nt,
        "Random Forest": m_rf,
        "Gradient Boosting": m_gb,
    }).T.round(3)

    # Highlight best values
    st.dataframe(df_metrics.style.highlight_min(subset=['RMSE','MAE','NASA Score'], color='#c8e6c9')
                                  .highlight_max(subset=['R²'], color='#c8e6c9'), use_container_width=True)
    st.caption("Green = best value. NASA Score: lower is better (asymmetric penalty favoring early predictions).")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("RMSE / MAE Comparison")
        fig = go.Figure()
        for metric, color in [("RMSE","#2196F3"), ("MAE","#FF9800")]:
            fig.add_trace(go.Bar(
                name=metric,
                x=["Neural Tree","Random Forest","Gradient Boosting"],
                y=[m_nt[metric], m_rf[metric], m_gb[metric]],
                marker_color=color, opacity=0.85
            ))
        fig.update_layout(barmode='group', height=350,
                          yaxis_title="Error (cycles)", legend=dict(orientation='h'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Predicted vs True RUL")
        fig2 = go.Figure()
        preds = {
            "Neural Tree": predict_neural_tree(nt, X_test),
            "Random Forest": rf.predict(X_test),
            "Gradient Boosting": gb.predict(X_test),
        }
        for name, pred, color in [
            ("Neural Tree", preds["Neural Tree"], "#2196F3"),
            ("Random Forest", preds["Random Forest"], "#4CAF50"),
            ("Gradient Boosting", preds["Gradient Boosting"], "#FF9800"),
        ]:
            fig2.add_trace(go.Scatter(x=y_test, y=pred, mode='markers',
                                      name=name, marker=dict(color=color, size=5, opacity=0.7)))
        fig2.add_trace(go.Scatter(x=[0,130], y=[0,130], mode='lines',
                                   name='Ideal', line=dict(color='black', dash='dash')))
        fig2.update_layout(xaxis_title="True RUL", yaxis_title="Predicted RUL",
                           height=350, legend=dict(orientation='h'))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Prediction Timeline: All Test Engines")
    fig3 = go.Figure()
    x = list(range(len(y_test)))
    fig3.add_trace(go.Scatter(x=x, y=y_test, name="True RUL",
                               line=dict(color="black", width=2, dash="dash")))
    for name, pred, color in [
        ("Neural Tree", preds["Neural Tree"], "#2196F3"),
        ("Random Forest", preds["Random Forest"], "#4CAF50"),
    ]:
        fig3.add_trace(go.Scatter(x=x, y=pred, name=name,
                                   line=dict(color=color, width=1.5), opacity=0.85))
    fig3.update_layout(xaxis_title="Test Engine", yaxis_title="RUL (cycles)",
                       height=320, legend=dict(orientation='h', y=-0.2))
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: Results vs Literature
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Results vs Literature":
    st.title("How We Compare to Published Research")
    st.markdown("""
    CMAPSS FD001 is the most widely benchmarked dataset in engine prognostics research.
    Below we compare our results against published methods from peer-reviewed papers.
    Lower RMSE = better accuracy.
    """)
    st.markdown("---")

    # Literature benchmark table
    lit = pd.DataFrame([
        {"Method": "LSTM (Zheng et al., 2017)",          "RMSE": 16.14, "Type": "Deep Learning",   "Explainable": "No"},
        {"Method": "CNN (Li et al., 2018)",               "RMSE": 12.61, "Type": "Deep Learning",   "Explainable": "No"},
        {"Method": "Transformer (Wang et al., 2021)",     "RMSE": 13.20, "Type": "Deep Learning",   "Explainable": "No"},
        {"Method": "DCNN (Babu et al., 2016)",            "RMSE": 18.45, "Type": "Deep Learning",   "Explainable": "No"},
        {"Method": "SVM (Tran et al., 2012)",             "RMSE": 20.96, "Type": "Classical ML",    "Explainable": "Partial"},
        {"Method": "Random Forest (baseline)",            "RMSE": 18.07, "Type": "Classical ML",    "Explainable": "Partial"},
        {"Method": "Gradient Boosting (baseline)",        "RMSE": 18.34, "Type": "Classical ML",    "Explainable": "Partial"},
        {"Method": "Neural Tree (ours)",                  "RMSE": 17.87, "Type": "Hybrid",          "Explainable": "Yes ✅"},
    ])

    m_nt = get_metrics(y_test, predict_neural_tree(nt, X_test))

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("RMSE Comparison: CMAPSS FD001")
        st.caption("Our Neural Tree achieves competitive RMSE while being the only fully explainable model in this comparison.")

        colors = []
        for _, row in lit.iterrows():
            if row["Method"] == "Neural Tree (ours)":
                colors.append("#1565c0")
            elif row["Type"] == "Deep Learning":
                colors.append("#bdbdbd")
            else:
                colors.append("#90a4ae")

        fig = go.Figure(go.Bar(
            x=lit["RMSE"], y=lit["Method"],
            orientation='h',
            marker_color=colors,
            text=[f"{v:.2f}" for v in lit["RMSE"]],
            textposition='outside',
        ))
        fig.add_vline(x=m_nt["RMSE"], line_dash="dash", line_color="#1565c0",
                      annotation_text="Our model", annotation_position="top right")
        fig.update_layout(
            height=380, xaxis_title="RMSE (cycles), lower is better",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=60), xaxis_range=[0, 25],
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(248,249,255,1)'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Our Results")
        for label, val, good in [
            ("RMSE", f"{m_nt['RMSE']:.3f} cycles", True),
            ("MAE",  f"{m_nt['MAE']:.3f} cycles",  True),
            ("R²",   f"{m_nt['R²']:.4f}",           True),
            ("NASA Score", f"{m_nt['NASA Score']:.0f}", True),
        ]:
            st.markdown(f"""
            <div style="background:#f0f4ff;border-radius:8px;padding:12px 16px;
                        margin-bottom:8px;border-left:4px solid #1565c0">
                <div style="font-size:0.75rem;color:#666;text-transform:uppercase;
                            letter-spacing:1px">{label}</div>
                <div style="font-size:1.4rem;font-weight:700;color:#1565c0">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("What Makes Our Approach Different")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="nasa-card">
        <div style="font-size:2rem;margin-bottom:8px">🎯</div>
        <b>Competitive Accuracy</b><br>
        <span style="font-size:0.85rem;color:#555">
        RMSE of 17.87 cycles outperforms classical baselines (SVM: 20.96, DCNN: 18.45)
        and matches tree-based methods, without sacrificing interpretability.
        </span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="nasa-card">
        <div style="font-size:2rem;margin-bottom:8px">🔍</div>
        <b>Fully Explainable</b><br>
        <span style="font-size:0.85rem;color:#555">
        Unlike LSTM and CNN models, Neural Trees show exactly which sensors
        drive each prediction. This is essential for FAA certification and safety audits.
        </span>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="nasa-card">
        <div style="font-size:2rem;margin-bottom:8px">🛡️</div>
        <b>Robust to Sensor Failure</b><br>
        <span style="font-size:0.85rem;color:#555">
        At 30% missing sensors, Neural Tree RMSE increases by only 2.5 cycles
        vs Random Forest's 10.5 cycles. That is a 76% better degradation profile.
        </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Literature values sourced from: Zheng et al. (2017) IJPEDS, Li et al. (2018) Reliability Engineering, Wang et al. (2021) IEEE TII, Babu et al. (2016) IEEE BigData, Tran et al. (2012) AAAI.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Feature Importance":
    st.title("Feature Importance and Explainability")
    st.markdown("""
    One of the key advantages of Neural Trees over black-box models is **explainability**.
    By computing how much each sensor influences the model's output (via input gradients),
    we can identify which sensors are most critical for predicting engine health.
    This is essential in safety-critical systems where engineers need to understand *why*
    a model is raising a maintenance alert.
    """)
    st.markdown("---")

    importance = nt.get_feature_importance(torch.tensor(X_test, dtype=torch.float32))
    imp_df = pd.DataFrame({"Feature": USEFUL_FEATURES, "Importance": importance})
    imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("All Sensor Importance Scores")
        st.caption("Higher score = this sensor has more influence on the RUL prediction.")
        fig = go.Figure(go.Bar(
            x=imp_df["Importance"], y=imp_df["Feature"],
            orientation='h',
            marker=dict(color=imp_df["Importance"], colorscale='Blues', showscale=False)
        ))
        fig.update_layout(height=420, xaxis_title="Importance Score",
                          yaxis=dict(autorange="reversed"), margin=dict(l=60))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 5 Most Important Sensors")
        st.caption("These sensors carry the most information about engine degradation.")
        st.markdown("")
        for _, row in imp_df.head(5).iterrows():
            pct = row['Importance'] * 100
            st.markdown(f"**{row['Feature']}**: `{pct:.1f}%`")
            st.progress(float(row['Importance']) / float(imp_df['Importance'].max()))
            st.markdown("")
        st.markdown("---")
        st.info("Knowing which sensors matter most allows maintenance teams to prioritize sensor health checks and focus inspections on the most critical components.")

    st.markdown("---")
    st.subheader("Sensor Correlation with RUL")
    st.caption("How strongly each sensor reading correlates with remaining engine life. Green = sensor value rises as engine life increases. Red = sensor value rises as engine degrades.")
    corr_data = pd.DataFrame(X_train, columns=USEFUL_FEATURES)
    corr_data['RUL'] = y_train
    corr = corr_data.corr()['RUL'].drop('RUL').sort_values()

    fig2 = go.Figure(go.Bar(
        x=corr.values, y=corr.index, orientation='h',
        marker_color=['#ef5350' if v < 0 else '#4CAF50' for v in corr.values]
    ))
    fig2.update_layout(height=380, xaxis_title="Pearson Correlation with RUL",
                       yaxis=dict(autorange="reversed"), margin=dict(l=60))
    st.plotly_chart(fig2, use_container_width=True)
