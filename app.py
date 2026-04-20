"""
Turbofan Engine RUL Prediction — Temporal Neural Tree Demo
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
    add_gaussian_noise, simulate_missing_sensors,
    load_cmapss_sequences
)
from src.neural_tree import NeuralTreeEnsemble, train_neural_tree, predict_neural_tree
from src.temporal_neural_tree import (TemporalNeuralTreeEnsemble,
                                       train_temporal_nt, predict_temporal_nt)
from src.lstm_baseline import (LSTMBaseline, train_lstm, predict_lstm,
                                apply_missing_to_sequences, apply_noise_to_sequences)
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Turbofan RUL — Temporal Neural Tree",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.hero-banner {
    background: linear-gradient(135deg, #0b0c2a 0%, #1a237e 50%, #0d47a1 100%);
    border-radius: 16px; padding: 40px 36px; color: white; margin-bottom: 24px;
}
.hero-banner h1 { font-size: 2rem; font-weight: 700; margin: 0 0 8px 0; color: white; }
.hero-banner p  { font-size: 1rem; opacity: 0.85; margin: 0; color: white; }
.badge {
    display: inline-block; background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3); border-radius: 20px;
    padding: 4px 14px; font-size: 0.78rem; margin: 6px 4px 0 0; color: white;
}
.nasa-card {
    background: #f8f9ff; border: 1px solid #e0e4f0;
    border-radius: 12px; padding: 20px; margin-bottom: 12px;
}
.stat-card {
    background: linear-gradient(135deg, #1a237e, #1565c0);
    border-radius: 12px; padding: 20px 16px; text-align: center; color: white;
}
.stat-card .num { font-size: 2.2rem; font-weight: 700; }
.stat-card .lbl { font-size: 0.8rem; opacity: 0.8; margin-top: 2px; }
.metric-card {
    background: #f0f4ff; border-radius: 10px; padding: 16px;
    text-align: center; border-left: 4px solid #2196F3;
}
.github-btn {
    background: #24292e; color: white !important; padding: 10px 20px;
    border-radius: 8px; text-decoration: none; font-weight: 600;
    display: inline-block; margin-top: 8px;
}
.section-label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; color: #5c6bc0; margin-bottom: 6px;
}
.unc-band { background: #e3f2fd; border-radius: 8px; padding: 12px 16px; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1)))

def get_metrics(y_true, y_pred):
    rmse = float(sqrt(mean_squared_error(y_true, y_pred)))
    return {"RMSE": rmse, "MAE": float(np.mean(np.abs(y_true-y_pred))),
            "R²": float(r2_score(y_true, y_pred)), "NASA": nasa_score(y_true, y_pred)}


# ── Model training (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_models_and_data():
    train_df, test_df = load_cmapss(dataset='FD001', data_dir='data')
    X_train, y_train, X_test, y_test, scaler = prepare_features(train_df, test_df)
    X_tr_seq, y_tr_seq, X_te_seq, y_te_seq, _ = \
        load_cmapss_sequences('FD001', 'data', seq_len=30)

    base = os.path.dirname(__file__)

    # Temporal Neural Tree — load saved weights or train
    tnt = TemporalNeuralTreeEnsemble(input_dim=17, hidden_dim=64,
                                      gru_layers=2, n_trees=5, depth=5)
    tnt_path = os.path.join(base, 'tnt_model.pt')
    if os.path.exists(tnt_path):
        tnt.load_state_dict(torch.load(tnt_path, map_location='cpu'))
    else:
        train_temporal_nt(tnt, X_tr_seq, y_tr_seq, epochs=200,
                          sensor_dropout=0.1, verbose=False)
        torch.save(tnt.state_dict(), tnt_path)
    tnt.eval()

    # Vanilla NT — load or train
    vnt = NeuralTreeEnsemble(input_dim=17, n_trees=5, depth=5, hidden_dim=32)
    vnt_path = os.path.join(base, 'vnt_model.pt')
    if os.path.exists(vnt_path):
        vnt.load_state_dict(torch.load(vnt_path, map_location='cpu'))
    else:
        train_neural_tree(vnt, X_train, y_train, epochs=200,
                          sensor_dropout=0.1, verbose=False)
        torch.save(vnt.state_dict(), vnt_path)
    vnt.eval()

    # LSTM — load or train
    lstm = LSTMBaseline(input_dim=17, hidden_dim=64, num_layers=2)
    lstm_path = os.path.join(base, 'lstm_model.pt')
    if os.path.exists(lstm_path):
        lstm.load_state_dict(torch.load(lstm_path, map_location='cpu'))
    else:
        train_lstm(lstm, X_tr_seq, y_tr_seq, epochs=150, verbose=False)
        torch.save(lstm.state_dict(), lstm_path)
    lstm.eval()

    # Ensemble baselines
    rf = RandomForestRegressor(n_estimators=200, max_depth=12,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    gb = HistGradientBoostingRegressor(max_iter=200, max_depth=5,
                                        learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)

    return (tnt, vnt, lstm, rf, gb,
            X_train, y_train, X_test, y_test,
            X_tr_seq, y_tr_seq, X_te_seq, y_te_seq,
            train_df, test_df, scaler)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="background:linear-gradient(135deg,#0b0c2a,#1a237e);border-radius:10px;
            padding:16px;margin-bottom:12px;text-align:center">
    <div style="font-size:2rem">✈️</div>
    <div style="color:white;font-weight:700;font-size:0.95rem;margin-top:4px">Turbofan RUL</div>
    <div style="color:rgba(255,255,255,0.7);font-size:0.75rem">Temporal Neural Tree</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", [
    "🏠 Overview",
    "🔮 RUL Prediction",
    "🌊 Robustness Test",
    "📊 Model Comparison",
    "🎯 Uncertainty",
    "🔍 Feature Importance"
])
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size:0.8rem;color:#555;line-height:1.8">
<b>Dataset</b><br>NASA CMAPSS FD001<br><br>
<b>Author</b><br>Cagri Temel / Hezarfen LLC<br><br>
<b>Model</b><br>Temporal Neural Tree<br>GRU + Soft Decision Trees<br>+ Leaf Uncertainty
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("[GitHub](https://github.com/cgrtml/turbofan-explainable-neural-trees)")

# ── Load ──────────────────────────────────────────────────────────────────────
(tnt, vnt, lstm, rf, gb,
 X_train, y_train, X_test, y_test,
 X_tr_seq, y_tr_seq, X_te_seq, y_te_seq,
 train_df, test_df, scaler) = load_models_and_data()

tnt_pred, tnt_std   = predict_temporal_nt(tnt, X_te_seq)
vnt_pred            = predict_neural_tree(vnt, X_test)
lstm_pred           = predict_lstm(lstm, X_te_seq)
rf_pred             = rf.predict(X_test)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class="hero-banner">
        <h1>Temporal Neural Trees for Turbofan Engine RUL Prediction</h1>
        <p>GRU sequence encoder with soft decision trees, calibrated uncertainty, and robustness to sensor failure.</p>
        <br>
        <span class="badge">GRU Encoder</span>
        <span class="badge">Soft Decision Trees</span>
        <span class="badge">Sensor Dropout</span>
        <span class="badge">Leaf Uncertainty</span>
        <span class="badge">NASA CMAPSS</span>
    </div>
    """, unsafe_allow_html=True)

    # NASA attribution
    st.markdown("""
    <div style="display:flex;align-items:center;gap:16px;background:#f0f4ff;
                border-radius:10px;padding:12px 20px;margin-bottom:8px;border:1px solid #dde3f5">
        <img src="https://www.nasa.gov/wp-content/themes/nasa/assets/images/nasa-logo.svg"
             height="48" style="flex-shrink:0"/>
        <div>
            <b style="font-size:0.95rem">Dataset: NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)</b><br>
            <span style="font-size:0.82rem;color:#555">
            100 turbofan engines run to failure under one operating condition (FD001 subset).
            Sensors: 21 channels measuring temperatures, pressures, fan speeds, and fuel flow.
            Ground-truth RUL labels provided by NASA Prognostics Center of Excellence.
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    m_tnt = get_metrics(y_te_seq, tnt_pred)
    m_lstm = get_metrics(y_te_seq, lstm_pred)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, num, lbl in [
        (c1, f"{m_tnt['RMSE']:.2f}", "TNT RMSE (cycles)"),
        (c2, f"{m_lstm['RMSE']:.2f}", "LSTM RMSE (cycles)"),
        (c3, f"{m_tnt['R²']:.3f}", "TNT R²"),
        (c4, "30%", "Sensor Failure"),
        (c5, "Yes", "Uncertainty Output"),
    ]:
        col.markdown(f"""
        <div class="stat-card">
            <div class="num">{num}</div>
            <div class="lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-label">Architecture</div>', unsafe_allow_html=True)
        st.subheader("Temporal Neural Tree")
        st.markdown("""
        **Three improvements over vanilla Neural Trees:**

        1. **GRU Encoder** processes 30-cycle sequences (same as LSTM), which closes the accuracy gap
        2. **Channel-level Sensor Dropout** zeros entire sensor channels during training,
           teaching the model to function without any given sensor
        3. **Leaf-level Gaussian Uncertainty:** each tree leaf outputs a mean and variance,
           combined into calibrated prediction intervals. No Monte Carlo sampling needed.

        **Key result:** At 30% missing sensors, TNT degrades only ~13% while LSTM degrades ~117%.
        """)
        st.markdown("""
        <div class="nasa-card" style="margin-top:12px">
        <b>Why not just use LSTM?</b><br>
        <span style="font-size:0.88rem;color:#444">
        LSTM is more accurate on clean data but catastrophically brittle when sensor channels
        fail completely. It interprets zeroed channels as extreme out-of-distribution readings,
        corrupting all subsequent hidden states. TNT's channel-level dropout training makes it
        immune to this failure mode. Additionally, LSTM provides no uncertainty estimates and no
        interpretable feature importance.
        </span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-label">Live Sensor Data</div>', unsafe_allow_html=True)
        st.subheader("Engine #1 Degradation")
        engine_1 = train_df[train_df['unit_id'] == 1].copy()
        fig = go.Figure()
        for sensor, color, label in [
            ('s4',  '#1565c0', 's4 — HPC Outlet Temp'),
            ('s11', '#2e7d32', 's11 — Bypass Ratio'),
            ('s12', '#e65100', 's12 — Burner Fuel-Air'),
        ]:
            vals = engine_1[sensor]
            norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
            fig.add_trace(go.Scatter(x=engine_1['time_cycle'], y=norm,
                                     name=label, line=dict(color=color, width=2)))
        fig.update_layout(xaxis_title="Flight Cycle", yaxis_title="Normalized Value",
                          height=300, margin=dict(t=10,b=10),
                          legend=dict(orientation='h', y=-0.3))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('[GitHub Repository](https://github.com/cgrtml/turbofan-explainable-neural-trees)')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: RUL Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 RUL Prediction":
    st.title("RUL Prediction with Uncertainty")

    st.info("""
**What is this page?**
This page shows real-time Remaining Useful Life (RUL) predictions for 100 turbofan engines
from the NASA CMAPSS FD001 test set.

- **RUL** = how many more operating cycles the engine can run before failure. 1 cycle ≈ one flight.
- **Engine 1–100** = individual test engines. Each has a known true RUL (ground truth) used to measure prediction accuracy.
- **Critical / Caution / Healthy** = fleet-level status based on predicted RUL thresholds (<40, 40–80, >80 cycles).
- **95% confidence interval** = TNT's uncertainty estimate. A wider band means the model is less certain (e.g. under sensor failure).
- **Noise σ / Missing sensors** = sliders to simulate degraded sensor conditions and see how each model responds.
    """)
    st.markdown("---")

    critical = int(np.sum(y_te_seq <= 40))
    caution  = int(np.sum((y_te_seq > 40) & (y_te_seq <= 80)))
    healthy  = int(np.sum(y_te_seq > 80))
    st.markdown(f"""
    <div style="background:#f8f9ff;border-radius:10px;padding:14px 20px;
                display:flex;gap:24px;margin-bottom:16px;border:1px solid #e0e4f0">
        <div><span style="font-size:0.72rem;font-weight:600;letter-spacing:1px;
                          text-transform:uppercase;color:#555">Fleet (100 engines)</span></div>
        <div><span style="color:#c62828;font-weight:700">{critical}</span>
             <span style="color:#888;font-size:0.85rem"> Critical (&lt;40)</span></div>
        <div><span style="color:#f57f17;font-weight:700">{caution}</span>
             <span style="color:#888;font-size:0.85rem"> Caution (40–80)</span></div>
        <div><span style="color:#2e7d32;font-weight:700">{healthy}</span>
             <span style="color:#888;font-size:0.85rem"> Healthy (&gt;80)</span></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        engine_id  = st.selectbox("Engine", list(range(1, 101)),
                                   format_func=lambda x: f"Engine {x} (RUL={int(y_te_seq[x-1])})")
        noise_level = st.slider("Noise σ", 0.0, 0.20, 0.0, 0.01)
        missing_pct = st.slider("Missing sensors %", 0, 50, 0, 10)

    # apply degradation
    X_seq_eng = X_te_seq[engine_id-1:engine_id].copy()
    X_snap_eng = X_test[engine_id-1:engine_id].copy()
    true_rul = float(y_te_seq[engine_id-1])

    if noise_level > 0:
        X_seq_eng  = apply_noise_to_sequences(X_seq_eng, noise_level)
        X_snap_eng = add_gaussian_noise(X_snap_eng, noise_level)
    if missing_pct > 0:
        X_seq_eng, _  = apply_missing_to_sequences(X_seq_eng, missing_pct/100)
        X_snap_eng, _ = simulate_missing_sensors(X_snap_eng, missing_pct/100)

    p_tnt, s_tnt = predict_temporal_nt(tnt, X_seq_eng)
    p_lstm        = predict_lstm(lstm, X_seq_eng)
    p_vnt         = predict_neural_tree(vnt, X_snap_eng)
    p_rf          = float(rf.predict(X_snap_eng)[0])

    status = "🔴 Critical" if true_rul <= 40 else ("🟡 Caution" if true_rul <= 80 else "🟢 Healthy")

    with col2:
        st.markdown(f"### Engine {engine_id} &nbsp; {status}")
        st.markdown(f"**True RUL: {true_rul:.0f} cycles**")

        c1, c2, c3, c4 = st.columns(4)
        for col, name, pred, color in [
            (c1, "Temporal NT", float(p_tnt[0]), "#1565c0"),
            (c2, "LSTM",        float(p_lstm[0]), "#455a64"),
            (c3, "Vanilla NT",  float(p_vnt[0]),  "#2e7d32"),
            (c4, "Rnd. Forest", p_rf,              "#e65100"),
        ]:
            err = abs(pred - true_rul)
            col.markdown(f"""
            <div class="metric-card" style="border-left-color:{color}">
                <h4 style="color:{color};margin:0;font-size:0.85rem">{name}</h4>
                <h2 style="margin:6px 0;font-size:1.6rem">{pred:.1f}</h2>
                <small>Error: {err:.1f} cycles</small>
            </div>""", unsafe_allow_html=True)

        # Uncertainty band for TNT
        lo = max(0, float(p_tnt[0]) - 2*float(s_tnt[0]))
        hi = float(p_tnt[0]) + 2*float(s_tnt[0])
        st.markdown(f"""
        <div class="unc-band">
            <b>TNT 95% confidence interval:</b> &nbsp;
            {lo:.1f} – {hi:.1f} cycles &nbsp;
            (predicted σ = {float(s_tnt[0]):.1f} cycles)
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Gauge for TNT with uncertainty
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=float(p_tnt[0]),
            title=dict(text="Temporal NT Prediction"),
            gauge=dict(
                axis=dict(range=[0, 130]),
                bar=dict(color="#1565c0"),
                steps=[dict(range=[0,40], color="#ffebee"),
                       dict(range=[40,80], color="#fff9c4"),
                       dict(range=[80,130], color="#e8f5e9")],
                threshold=dict(line=dict(color="black", width=3), value=true_rul)
            ),
            number=dict(suffix=" cycles")
        ))
        fig.update_layout(height=240, margin=dict(t=30,b=10,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Robustness Test
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌊 Robustness Test":
    st.title("Robustness: TNT vs LSTM vs Vanilla NT")
    st.markdown("""
    This page demonstrates the core finding of the paper: **sensor dropout augmentation
    and temporal averaging are distinct robustness properties**. LSTM is more resistant
    to Gaussian noise; TNT is dramatically more resistant to complete sensor failure.
    """)

    tab1, tab2 = st.tabs(["Missing Sensors", "Gaussian Noise"])

    with tab1:
        st.subheader("RMSE vs Missing Sensor Channels")
        st.caption("Entire channels zeroed; simulates complete sensor failure.")

        miss_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        tnt_m, lstm_m, vnt_m, rf_m = [], [], [], []

        with st.spinner("Computing..."):
            for r in miss_ratios:
                Xsm, _ = apply_missing_to_sequences(X_te_seq, r) if r > 0 \
                          else (X_te_seq.copy(), None)
                Xm, _  = simulate_missing_sensors(X_test, r) if r > 0 \
                          else (X_test.copy(), None)
                p_t, _ = predict_temporal_nt(tnt, Xsm)
                tnt_m.append(float(sqrt(mean_squared_error(y_te_seq, p_t))))
                lstm_m.append(float(sqrt(mean_squared_error(y_te_seq, predict_lstm(lstm, Xsm)))))
                vnt_m.append(float(sqrt(mean_squared_error(y_test, predict_neural_tree(vnt, Xm)))))
                rf_m.append(float(sqrt(mean_squared_error(y_test, rf.predict(Xm)))))

        fig = go.Figure()
        for name, vals, color, dash in [
            ("Temporal NT (ours)", tnt_m,  "#1565c0", "solid"),
            ("Vanilla NT",         vnt_m,  "#2e7d32", "dash"),
            ("LSTM (no dropout)",  lstm_m, "#455a64", "dashdot"),
            ("Random Forest",      rf_m,   "#bdbdbd", "dot"),
        ]:
            fig.add_trace(go.Scatter(x=[r*100 for r in miss_ratios], y=vals,
                                     name=name, line=dict(color=color, width=2.5, dash=dash),
                                     mode='lines+markers', marker=dict(size=8)))
        fig.update_layout(xaxis_title="Missing Channels (%)",
                          yaxis_title="RMSE (cycles)", height=380,
                          legend=dict(orientation='h', y=-0.22))
        st.plotly_chart(fig, use_container_width=True)

        delta_tnt  = tnt_m[3]  - tnt_m[0]
        delta_lstm = lstm_m[3] - lstm_m[0]
        delta_vnt  = vnt_m[3]  - vnt_m[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("TNT at 30% missing",  f"{tnt_m[3]:.2f}",  f"+{delta_tnt:.2f} from clean")
        c2.metric("LSTM at 30% missing", f"{lstm_m[3]:.2f}", f"+{delta_lstm:.2f} from clean")
        c3.metric("VanillaNT at 30%",    f"{vnt_m[3]:.2f}",  f"+{delta_vnt:.2f} from clean")
        st.success(f"At 30% missing sensors: TNT degrades +{delta_tnt:.1f} cycles "
                   f"vs LSTM +{delta_lstm:.1f} cycles. Sensor dropout is the key.")

    with tab2:
        st.subheader("RMSE vs Gaussian Noise Level")
        st.caption("Independent per-timestep noise. LSTM's temporal averaging helps here.")

        sigmas = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
        tnt_n, lstm_n, vnt_n, rf_n = [], [], [], []

        with st.spinner("Computing..."):
            for s in sigmas:
                Xsn = apply_noise_to_sequences(X_te_seq, s) if s > 0 else X_te_seq.copy()
                Xn  = add_gaussian_noise(X_test, s) if s > 0 else X_test.copy()
                p_t, _ = predict_temporal_nt(tnt, Xsn)
                tnt_n.append(float(sqrt(mean_squared_error(y_te_seq, p_t))))
                lstm_n.append(float(sqrt(mean_squared_error(y_te_seq, predict_lstm(lstm, Xsn)))))
                vnt_n.append(float(sqrt(mean_squared_error(y_test, predict_neural_tree(vnt, Xn)))))
                rf_n.append(float(sqrt(mean_squared_error(y_test, rf.predict(Xn)))))

        fig2 = go.Figure()
        for name, vals, color, dash in [
            ("Temporal NT (ours)", tnt_n,  "#1565c0", "solid"),
            ("LSTM (no dropout)",  lstm_n, "#455a64", "dashdot"),
            ("Vanilla NT",         vnt_n,  "#2e7d32", "dash"),
            ("Random Forest",      rf_n,   "#bdbdbd", "dot"),
        ]:
            fig2.add_trace(go.Scatter(x=sigmas, y=vals, name=name,
                                      line=dict(color=color, width=2.5, dash=dash),
                                      mode='lines+markers', marker=dict(size=8)))
        fig2.update_layout(xaxis_title="Noise σ", yaxis_title="RMSE (cycles)",
                           height=380, legend=dict(orientation='h', y=-0.22))
        st.plotly_chart(fig2, use_container_width=True)
        st.info("LSTM is more noise-robust because temporal averaging across 30 timesteps "
                "attenuates independent per-timestep noise. TNT follows a similar trend. "
                "At realistic noise (σ ≤ 0.05) all models stay within 3 cycles of clean-data performance.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("Model Performance Comparison")
    st.markdown("---")

    m_tnt  = get_metrics(y_te_seq, tnt_pred)
    m_lstm = get_metrics(y_te_seq, lstm_pred)
    m_vnt  = get_metrics(y_test,   vnt_pred)
    m_rf   = get_metrics(y_test,   rf_pred)

    df_m = pd.DataFrame({
        "Temporal NT (ours)": m_tnt,
        "LSTM":               m_lstm,
        "Vanilla NT":         m_vnt,
        "Random Forest":      m_rf,
    }).T.round(3)

    st.subheader("Baseline Metrics on CMAPSS FD001")
    st.dataframe(
        df_m.style
            .highlight_min(subset=['RMSE','MAE','NASA'], color='#c8e6c9')
            .highlight_max(subset=['R²'],                color='#c8e6c9'),
        use_container_width=True
    )
    st.caption("Sequence models (TNT, LSTM) use 30-cycle input; snapshot models use single-cycle input.")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("RMSE Comparison")
        models = ["Temporal NT", "LSTM", "Vanilla NT", "Rnd. Forest"]
        rmses  = [m_tnt['RMSE'], m_lstm['RMSE'], m_vnt['RMSE'], m_rf['RMSE']]
        colors = ["#1565c0", "#455a64", "#2e7d32", "#bdbdbd"]
        fig = go.Figure(go.Bar(x=models, y=rmses, marker_color=colors,
                               text=[f"{v:.2f}" for v in rmses], textposition='outside'))
        fig.update_layout(yaxis_title="RMSE (cycles)", height=320,
                          yaxis=dict(range=[0, max(rmses)*1.2]))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("True vs Predicted RUL")
        fig2 = go.Figure()
        for name, preds, yt, color in [
            ("Temporal NT", tnt_pred, y_te_seq, "#1565c0"),
            ("LSTM",        lstm_pred, y_te_seq, "#455a64"),
        ]:
            fig2.add_trace(go.Scatter(x=yt, y=preds, mode='markers', name=name,
                                      marker=dict(color=color, size=5, opacity=0.65)))
        fig2.add_trace(go.Scatter(x=[0,130], y=[0,130], mode='lines',
                                   name='Ideal', line=dict(color='black', dash='dash')))
        fig2.update_layout(xaxis_title="True RUL", yaxis_title="Predicted RUL",
                           height=320, legend=dict(orientation='h'))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Capability Summary")
    cap_df = pd.DataFrame({
        "Model":               ["Temporal NT (ours)", "LSTM", "Vanilla NT", "Random Forest"],
        "Uses Sequences":      ["Yes", "Yes", "No",  "No"],
        "Sensor Dropout":      ["Yes", "No",  "Yes", "No"],
        "Uncertainty Output":  ["Yes", "No",  "No",  "No"],
        "Interpretable":       ["Yes", "No",  "Yes", "Partial"],
        "RMSE (clean)":        [f"{m_tnt['RMSE']:.2f}", f"{m_lstm['RMSE']:.2f}",
                                 f"{m_vnt['RMSE']:.2f}", f"{m_rf['RMSE']:.2f}"],
    })
    st.dataframe(cap_df.set_index("Model"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: Uncertainty
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Uncertainty":
    st.title("Calibrated Uncertainty Quantification")
    st.markdown("""
    The Temporal Neural Tree outputs a **prediction interval** for every engine, derived from
    leaf-level Gaussian parameters trained with negative log-likelihood. A well-calibrated model's
    predicted standard deviation should track its actual prediction error.
    """)
    st.markdown("---")

    errors = np.abs(tnt_pred - y_te_seq)
    bins = np.percentile(tnt_std, np.linspace(0, 100, 11))
    bin_idx = np.digitize(tnt_std, bins[1:-1])
    bin_stds, bin_errs, bin_ns = [], [], []
    for b in range(10):
        mask = bin_idx == b
        if mask.sum() > 2:
            bin_stds.append(float(tnt_std[mask].mean()))
            bin_errs.append(float(errors[mask].mean()))
            bin_ns.append(int(mask.sum()))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Calibration: Predicted σ vs Actual Error")
        lim = max(max(bin_stds), max(bin_errs)) * 1.15
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode='lines', name='Perfect',
                                  line=dict(color='black', dash='dash', width=1)))
        fig.add_trace(go.Scatter(x=bin_stds, y=bin_errs, mode='markers+lines',
                                  name='TNT', marker=dict(color='#1565c0', size=9),
                                  line=dict(color='#1565c0', width=2),
                                  text=[f"n={n}" for n in bin_ns]))
        fig.update_layout(xaxis_title="Predicted σ (cycles)",
                          yaxis_title="Mean absolute error (cycles)",
                          height=340, legend=dict(orientation='h'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Prediction Intervals (100 Engines)")
        sorted_idx = np.argsort(y_te_seq)
        y_sorted   = y_te_seq[sorted_idx]
        mu_sorted  = tnt_pred[sorted_idx]
        lo_sorted  = np.maximum(0, mu_sorted - 2*tnt_std[sorted_idx])
        hi_sorted  = mu_sorted + 2*tnt_std[sorted_idx]
        x_ax = list(range(len(y_sorted)))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=x_ax + x_ax[::-1],
            y=list(hi_sorted) + list(lo_sorted[::-1]),
            fill='toself', fillcolor='rgba(21,101,192,0.15)',
            line=dict(color='rgba(255,255,255,0)'), name='95% CI'
        ))
        fig2.add_trace(go.Scatter(x=x_ax, y=mu_sorted, mode='lines',
                                   name='Predicted', line=dict(color='#1565c0', width=1.5)))
        fig2.add_trace(go.Scatter(x=x_ax, y=y_sorted, mode='markers',
                                   name='True RUL', marker=dict(color='black', size=4)))
        fig2.update_layout(xaxis_title="Engine (sorted by true RUL)",
                           yaxis_title="RUL (cycles)", height=340,
                           legend=dict(orientation='h'))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    coverage = float(np.mean((y_te_seq >= tnt_pred - 2*tnt_std) &
                              (y_te_seq <= tnt_pred + 2*tnt_std)) * 100)
    c1, c2, c3 = st.columns(3)
    c1.metric("95% CI Coverage",  f"{coverage:.1f}%", "target: 95%")
    c2.metric("Mean predicted σ", f"{tnt_std.mean():.1f} cycles")
    c3.metric("Mean actual error",f"{errors.mean():.1f} cycles")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Feature Importance":
    st.title("Gradient-Based Feature Importance")
    st.markdown("""
    Importance is computed by backpropagating through the full GRU encoder + tree ensemble
    and measuring mean absolute gradient with respect to each input channel, averaged over
    all timesteps. No post-hoc approximation; this reflects the model's actual sensitivity.
    """)
    st.markdown("---")

    importance = tnt.get_feature_importance(torch.tensor(X_te_seq, dtype=torch.float32))
    imp_df = pd.DataFrame({"Feature": USEFUL_FEATURES, "Importance": importance})
    imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure(go.Bar(
            x=imp_df["Importance"] * 100, y=imp_df["Feature"], orientation='h',
            marker=dict(color=imp_df["Importance"]*100, colorscale='Blues', showscale=False)
        ))
        fig.update_layout(height=420, xaxis_title="Importance (%)",
                          yaxis=dict(autorange="reversed"), margin=dict(l=60))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 5 Sensors")
        for _, row in imp_df.head(5).iterrows():
            pct = row['Importance'] * 100
            st.markdown(f"**{row['Feature']}**: `{pct:.1f}%`")
            st.progress(float(row['Importance']) / float(imp_df['Importance'].max()))
        st.markdown("---")
        st.info("High-pressure turbine sensors (s11, s4, s12) dominate, consistent with "
                "the known CMAPSS fault mode where HPT components degrade first.")

    st.markdown("---")
    st.subheader("Sensor Correlation with RUL")
    corr_data = pd.DataFrame(X_train, columns=USEFUL_FEATURES)
    corr_data['RUL'] = y_train
    corr = corr_data.corr()['RUL'].drop('RUL').sort_values()
    fig2 = go.Figure(go.Bar(
        x=corr.values, y=corr.index, orientation='h',
        marker_color=['#ef5350' if v < 0 else '#4CAF50' for v in corr.values]
    ))
    fig2.update_layout(height=380, xaxis_title="Pearson r with RUL",
                       yaxis=dict(autorange="reversed"), margin=dict(l=60))
    st.plotly_chart(fig2, use_container_width=True)
