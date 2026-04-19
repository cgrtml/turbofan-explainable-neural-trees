"""
Full experiment runner — NASA CMAPSS Turbofan RUL Prediction
Experiments: Baseline, Noise Robustness, Missing Sensor, Multi-Dataset
Now includes LSTM baseline for fair deep-learning comparison.
"""

import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from src.data_preprocessing import (
    load_cmapss, prepare_features, USEFUL_FEATURES,
    add_gaussian_noise, simulate_missing_sensors,
    load_cmapss_sequences
)
from src.neural_tree import NeuralTreeEnsemble, train_neural_tree, predict_neural_tree
from src.lstm_baseline import (
    LSTMBaseline, train_lstm, predict_lstm,
    apply_missing_to_sequences, apply_noise_to_sequences
)
from src.visualization import (
    plot_rul_comparison, plot_metrics_bar, plot_noise_robustness,
    plot_missing_sensor_robustness, plot_feature_importance,
    plot_training_loss, plot_rul_scatter
)
import torch
import matplotlib.pyplot as plt

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

SEQ_LEN = 30


# ── Helpers ───────────────────────────────────────────────────────────────────

def _plot_robustness_4models(df, x_col, xlabel, title, fname, scale_x=1):
    """4-model robustness line plot (RMSE, MAE, R²)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    styles = [
        ("Neural Tree",       "NT",   "#000000", "o-"),
        ("LSTM",              "LSTM", "#444444", "D-."),
        ("Random Forest",     "RF",   "#777777", "s--"),
        ("Gradient Boosting", "XGB",  "#aaaaaa", "^:"),
    ]
    x = df[x_col] * scale_x
    for ax, metric, ylabel in zip(axes,
                                   ['RMSE', 'MAE', 'R2'],
                                   ['RMSE (cycles)', 'MAE (cycles)', 'R²']):
        for name, prefix, color, marker in styles:
            col = f"{prefix}_{metric}"
            if col in df.columns:
                ax.plot(x, df[col], marker, color=color,
                        label=name, linewidth=2, markersize=6)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs {xlabel}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f'results/figures/{fname}'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def compute_metrics(y_true, y_pred):
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    r2    = float(r2_score(y_true, y_pred))
    score = float(nasa_score(y_true, y_pred))
    return {"RMSE": round(rmse, 3), "MAE": round(mae, 3),
            "R2": round(r2, 4), "NASA_Score": round(score, 1)}


def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    return np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))


def build_rf():
    return RandomForestRegressor(n_estimators=200, max_depth=12,
                                 random_state=42, n_jobs=4)

def build_gb():
    return HistGradientBoostingRegressor(max_iter=200, max_depth=5,
                                         learning_rate=0.1, random_state=42)

def build_nt(input_dim):
    return NeuralTreeEnsemble(input_dim=input_dim, n_trees=5, depth=5,
                               hidden_dim=32, dropout=0.15)

def build_lstm(input_dim):
    return LSTMBaseline(input_dim=input_dim, hidden_dim=64,
                        num_layers=2, dropout=0.2)


# ── Load snapshot data (Neural Tree, RF, GB) ─────────────────────────────────

print("=" * 65)
print("Loading CMAPSS FD001 — snapshot features...")
print("=" * 65)
train_df, test_df = load_cmapss(dataset='FD001', data_dir='data')
X_train, y_train, X_test, y_test, scaler = prepare_features(train_df, test_df)
INPUT_DIM = X_train.shape[1]
print(f"Snapshot: X_train={X_train.shape} | X_test={X_test.shape}")

# ── Load sequence data (LSTM) ─────────────────────────────────────────────────

print("\nLoading CMAPSS FD001 — sequences (seq_len=30)...")
Xs_train, ys_train, Xs_test, ys_test, _ = load_cmapss_sequences(
    dataset='FD001', data_dir='data', seq_len=SEQ_LEN)
print(f"Sequences: X_train={Xs_train.shape} | X_test={Xs_test.shape}")


# ── EXPERIMENT 1: Baseline Comparison ────────────────────────────────────────
print("\n" + "=" * 65)
print("EXPERIMENT 1: Baseline RUL Prediction (FD001)")
print("=" * 65)

# Neural Tree
print("\n[1/4] Training Neural Tree Ensemble...")
nt_model = build_nt(INPUT_DIM)
nt_history = train_neural_tree(nt_model, X_train, y_train,
                                epochs=200, lr=5e-4, sensor_dropout=0.1,
                                verbose=True)
y_pred_nt = predict_neural_tree(nt_model, X_test)
nt_m = compute_metrics(y_test, y_pred_nt)
print(f"  Neural Tree -> {nt_m}")

# LSTM
print("\n[2/4] Training LSTM Baseline (seq_len=30, no sensor dropout)...")
lstm_model = build_lstm(INPUT_DIM)
lstm_history = train_lstm(lstm_model, Xs_train, ys_train,
                           epochs=150, lr=1e-3, verbose=True)
y_pred_lstm = predict_lstm(lstm_model, Xs_test)
lstm_m = compute_metrics(ys_test, y_pred_lstm)
print(f"  LSTM        -> {lstm_m}")

# Random Forest
print("\n[3/4] Training Random Forest...")
rf_model = build_rf()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_m = compute_metrics(y_test, y_pred_rf)
print(f"  RF          -> {rf_m}")

# Gradient Boosting
print("\n[4/4] Training Gradient Boosting...")
gb_model = build_gb()
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
gb_m = compute_metrics(y_test, y_pred_gb)
print(f"  GB          -> {gb_m}")

results = {
    "Neural Tree":       {**nt_m,   "predictions": y_pred_nt,   "history": nt_history},
    "LSTM":              {**lstm_m, "predictions": y_pred_lstm},
    "Random Forest":     {**rf_m,   "predictions": y_pred_rf},
    "Gradient Boosting": {**gb_m,   "predictions": y_pred_gb},
}

print("\nGenerating Experiment 1 figures...")
plot_rul_comparison(y_test, {k: v for k, v in results.items()
                             if k != "LSTM"}, max_samples=100)
plot_metrics_bar(results)
plot_rul_scatter(y_test, {k: v for k, v in results.items() if k != "LSTM"})
plot_training_loss(nt_history)

pd.DataFrame({
    name: {k: v for k, v in m.items() if k in ['RMSE','MAE','R2','NASA_Score']}
    for name, m in results.items()
}).T.to_csv('results/table1_baseline.csv')


# ── EXPERIMENT 2: Sensor Noise Robustness ─────────────────────────────────────
print("\n" + "=" * 65)
print("EXPERIMENT 2: Sensor Noise Robustness")
print("=" * 65)

noise_levels = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
noise_rows = []
for std in noise_levels:
    X_n    = add_gaussian_noise(X_test, noise_std=std)  if std > 0 else X_test.copy()
    Xs_n   = apply_noise_to_sequences(Xs_test, std)     if std > 0 else Xs_test.copy()

    row = {"noise_std": std}
    nt_n   = compute_metrics(y_test,  predict_neural_tree(nt_model, X_n))
    lstm_n = compute_metrics(ys_test, predict_lstm(lstm_model, Xs_n))
    rf_n   = compute_metrics(y_test,  rf_model.predict(X_n))
    gb_n   = compute_metrics(y_test,  gb_model.predict(X_n))
    for k in ['RMSE','MAE','R2']:
        row[f"NT_{k}"]   = nt_n[k]
        row[f"LSTM_{k}"] = lstm_n[k]
        row[f"RF_{k}"]   = rf_n[k]
        row[f"XGB_{k}"]  = gb_n[k]
    noise_rows.append(row)
    print(f"  σ={std:.2f} | NT={row['NT_RMSE']:.3f} | "
          f"LSTM={row['LSTM_RMSE']:.3f} | RF={row['RF_RMSE']:.3f} | "
          f"GB={row['XGB_RMSE']:.3f}")

noise_df = pd.DataFrame(noise_rows)
_plot_robustness_4models(noise_df, x_col='noise_std',
                          xlabel='Noise Level (σ)',
                          title='Robustness Under Sensor Noise (Gaussian σ)',
                          fname='fig3_noise_robustness.png')
noise_df.to_csv('results/table2_noise_robustness.csv', index=False)


# ── EXPERIMENT 3: Missing Sensor Robustness ───────────────────────────────────
print("\n" + "=" * 65)
print("EXPERIMENT 3: Missing Sensor Scenario")
print("=" * 65)

missing_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
miss_rows = []
for ratio in missing_ratios:
    if ratio > 0:
        X_m,  dropped  = simulate_missing_sensors(X_test,  missing_ratio=ratio)
        Xs_m, dropped2 = apply_missing_to_sequences(Xs_test, missing_ratio=ratio)
    else:
        X_m,  dropped  = X_test.copy(),  []
        Xs_m, dropped2 = Xs_test.copy(), []

    row = {"missing_ratio": ratio, "n_dropped": len(dropped)}
    nt_n   = compute_metrics(y_test,  predict_neural_tree(nt_model, X_m))
    lstm_n = compute_metrics(ys_test, predict_lstm(lstm_model, Xs_m))
    rf_n   = compute_metrics(y_test,  rf_model.predict(X_m))
    gb_n   = compute_metrics(y_test,  gb_model.predict(X_m))
    for k in ['RMSE','MAE','R2']:
        row[f"NT_{k}"]   = nt_n[k]
        row[f"LSTM_{k}"] = lstm_n[k]
        row[f"RF_{k}"]   = rf_n[k]
        row[f"XGB_{k}"]  = gb_n[k]
    miss_rows.append(row)
    print(f"  missing={int(ratio*100):2d}% | NT={row['NT_RMSE']:.3f} | "
          f"LSTM={row['LSTM_RMSE']:.3f} | RF={row['RF_RMSE']:.3f} | "
          f"GB={row['XGB_RMSE']:.3f}")

missing_df = pd.DataFrame(miss_rows)
_plot_robustness_4models(missing_df, x_col='missing_ratio',
                          xlabel='Missing Sensors (%)', scale_x=100,
                          title='Robustness Under Missing Sensor Channels',
                          fname='fig4_missing_sensor_robustness.png')
missing_df.to_csv('results/table3_missing_sensor.csv', index=False)


# ── EXPERIMENT 4: Multi-Dataset Generalization ────────────────────────────────
print("\n" + "=" * 65)
print("EXPERIMENT 4: Multi-Dataset Generalization (FD001–FD004)")
print("=" * 65)

multi_rows = []
for ds in ['FD001', 'FD002', 'FD003', 'FD004']:
    print(f"\n  Dataset: {ds}")
    tr, te = load_cmapss(dataset=ds, data_dir='data')
    Xtr, ytr, Xte, yte, _ = prepare_features(tr, te)
    idim = Xtr.shape[1]

    Xs_tr, ys_tr, Xs_te, ys_te, _ = load_cmapss_sequences(
        dataset=ds, data_dir='data', seq_len=SEQ_LEN)

    # Neural Tree
    m_nt = build_nt(idim)
    train_neural_tree(m_nt, Xtr, ytr, epochs=200, lr=5e-4,
                      sensor_dropout=0.1, verbose=False)
    nt_multi = compute_metrics(yte, predict_neural_tree(m_nt, Xte))

    # LSTM
    m_lstm = build_lstm(idim)
    train_lstm(m_lstm, Xs_tr, ys_tr, epochs=150, lr=1e-3, verbose=False)
    lstm_multi = compute_metrics(ys_te, predict_lstm(m_lstm, Xs_te))

    # RF
    m_rf = build_rf(); m_rf.fit(Xtr, ytr)
    rf_multi = compute_metrics(yte, m_rf.predict(Xte))

    # GB
    m_gb = build_gb(); m_gb.fit(Xtr, ytr)
    gb_multi = compute_metrics(yte, m_gb.predict(Xte))

    row = {"Dataset": ds,
           "NT_RMSE":   nt_multi["RMSE"],   "NT_R2":   nt_multi["R2"],
           "LSTM_RMSE": lstm_multi["RMSE"], "LSTM_R2": lstm_multi["R2"],
           "RF_RMSE":   rf_multi["RMSE"],   "RF_R2":   rf_multi["R2"],
           "GB_RMSE":   gb_multi["RMSE"],   "GB_R2":   gb_multi["R2"]}
    multi_rows.append(row)
    print(f"    NT: {nt_multi['RMSE']:.3f} | LSTM: {lstm_multi['RMSE']:.3f} | "
          f"RF: {rf_multi['RMSE']:.3f} | GB: {gb_multi['RMSE']:.3f}")

multi_df = pd.DataFrame(multi_rows)
multi_df.to_csv('results/table4_multi_dataset.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Multi-Dataset Generalization (FD001–FD004)", fontsize=13, fontweight='bold')
x  = np.arange(4)
w  = 0.20
for ax, metric, ylabel in zip(axes, ['RMSE', 'R2'], ['RMSE (cycles)', 'R²']):
    ax.bar(x - 1.5*w, multi_df[f'NT_{metric}'],   w, label='Neural Tree',       color='#333333', alpha=0.85)
    ax.bar(x - 0.5*w, multi_df[f'LSTM_{metric}'], w, label='LSTM',              color='#666666', alpha=0.85)
    ax.bar(x + 0.5*w, multi_df[f'RF_{metric}'],   w, label='Random Forest',     color='#999999', alpha=0.85)
    ax.bar(x + 1.5*w, multi_df[f'GB_{metric}'],   w, label='Gradient Boosting', color='#cccccc', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(['FD001','FD002','FD003','FD004'])
    ax.set_ylabel(ylabel); ax.set_title(f'{ylabel} by Dataset')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig('results/figures/fig8_multi_dataset.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved: results/figures/fig8_multi_dataset.png")


# ── Feature Importance ────────────────────────────────────────────────────────
importance = nt_model.get_feature_importance(
    torch.tensor(X_test[:200], dtype=torch.float32)
)
plot_feature_importance(importance, USEFUL_FEATURES, top_n=10)


# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TABLE I — Baseline (FD001)")
print("=" * 65)
print(f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'NASA_Score':>12}")
print("-" * 60)
for name, m in results.items():
    print(f"{name:<22} {m['RMSE']:>8.3f} {m['MAE']:>8.3f} "
          f"{m['R2']:>8.4f} {m['NASA_Score']:>12.1f}")

print("\n" + "=" * 65)
print("ROBUSTNESS SUMMARY — 30% Missing Sensors")
print("=" * 65)
r30 = missing_df[missing_df['missing_ratio'] == 0.3].iloc[0]
r00 = missing_df[missing_df['missing_ratio'] == 0.0].iloc[0]
for model, col in [("Neural Tree","NT"),("LSTM","LSTM"),("RF","RF"),("GB","XGB")]:
    delta = r30[f"{col}_RMSE"] - r00[f"{col}_RMSE"]
    print(f"  {model:<20} baseline={r00[f'{col}_RMSE']:.3f}  "
          f"at_30%={r30[f'{col}_RMSE']:.3f}  delta=+{delta:.3f}")

print("\n" + "=" * 65)
print("TABLE IV — Multi-Dataset Generalization")
print("=" * 65)
print(multi_df.to_string(index=False))

print("\nAll results saved to results/")
print("Figures saved to results/figures/")
print("\nDONE!")
