"""
Regenerates all paper figures in color from saved results_full.json.
Does NOT retrain models. Run after run_full_experiments.py has completed.
"""
import sys
sys.path.insert(0, 'src')

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from math import sqrt

os.makedirs('figures', exist_ok=True)

# ── load results ──────────────────────────────────────────────────────────────
with open('results_full.json') as f:
    res = json.load(f)

baseline   = res['baseline']
miss_res   = res['missing_sensor']
noise_res  = res['noise']
miss_ratios = miss_res['ratios']
sigmas      = noise_res['sigmas']

def rmse(yt, yp): return float(sqrt(np.mean((np.array(yt) - np.array(yp)) ** 2)))
def r2(yt, yp):
    yt, yp = np.array(yt), np.array(yp)
    return 1 - np.sum((yt - yp)**2) / np.sum((yt - np.mean(yt))**2)

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9,
    'legend.fontsize': 7.5, 'lines.linewidth': 1.8,
    'lines.markersize': 5.0, 'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'TemporalNT': '#1f77b4',   # blue
    'VanillaNT':  '#ff7f0e',   # orange
    'LSTM':       '#2ca02c',   # green
    'RF':         '#d62728',   # red
    'GB':         '#9467bd',   # purple
}
MARKS = {'TemporalNT': 'o', 'VanillaNT': 's', 'LSTM': '^', 'RF': 'D', 'GB': 'v'}
LS    = {'TemporalNT': '-', 'VanillaNT': '--', 'LSTM': '-.', 'RF': ':', 'GB': ':'}
LABELS = {
    'TemporalNT': 'Temporal NT (ours)',
    'VanillaNT':  'Vanilla NT',
    'LSTM':       'LSTM',
    'RF':         'RF',
    'GB':         'GB',
}

# ── Fig 4: missing sensor robustness ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.8, 2.9))
x_pct = [r * 100 for r in miss_ratios]
for n in ['TemporalNT', 'VanillaNT', 'LSTM', 'RF']:
    ax.plot(x_pct, miss_res[n], color=COLORS[n], marker=MARKS[n],
            ls=LS[n], label=LABELS[n], zorder=3 if 'NT' in n else 2)
ax.set_xlabel('Missing sensor channels (%)')
ax.set_ylabel('RMSE (cycles)')
ax.set_title('Missing Sensor Robustness')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.25, linewidth=0.5)
plt.tight_layout()
plt.savefig('figures/fig4_missing_sensor_robustness.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig4_missing_sensor_robustness.png")

# ── Fig 3: noise robustness ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.8, 2.9))
for n in ['TemporalNT', 'VanillaNT', 'LSTM', 'RF']:
    ax.plot(sigmas, noise_res[n], color=COLORS[n], marker=MARKS[n],
            ls=LS[n], label=LABELS[n], zorder=3 if 'NT' in n else 2)
ax.set_xlabel('Gaussian noise std $\\sigma$')
ax.set_ylabel('RMSE (cycles)')
ax.set_title('Noise Robustness')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.25, linewidth=0.5)
plt.tight_layout()
plt.savefig('figures/fig3_noise_robustness.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig3_noise_robustness.png")

# ── Fig 5: feature importance (need model) ────────────────────────────────────
# Load TNT model and recompute feature importance
try:
    from data_preprocessing import load_cmapss_sequences, USEFUL_FEATURES
    from temporal_neural_tree import (TemporalNeuralTreeEnsemble,
                                       predict_temporal_nt)

    _, _, X_te_seq, y_te_seq, _ = load_cmapss_sequences('FD001', 'data', seq_len=30)

    tnt = TemporalNeuralTreeEnsemble(input_dim=17, hidden_dim=64,
                                      gru_layers=2, n_trees=5, depth=5)
    tnt.load_state_dict(torch.load('tnt_model.pt', map_location='cpu'))
    tnt.eval()

    feat_imp = tnt.get_feature_importance(
        torch.tensor(X_te_seq, dtype=torch.float32))
    feat_names = USEFUL_FEATURES
    top_idx = np.argsort(feat_imp)[::-1][:10]
    top_names = [feat_names[i] for i in top_idx]
    top_vals  = [feat_imp[i] * 100 for i in top_idx]

    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    bar_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_names)))
    ax.barh(range(len(top_names)), top_vals[::-1], color=bar_colors[::-1])
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel('Importance (%)')
    ax.set_title('Top-10 Feature Importance')
    ax.grid(True, alpha=0.25, linewidth=0.5, axis='x')
    plt.tight_layout()
    plt.savefig('figures/fig5_feature_importance.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved fig5_feature_importance.png")

    # ── Fig: uncertainty calibration ────────────────────────────────────────────
    tnt_pred, tnt_std = predict_temporal_nt(tnt, X_te_seq)
    errors   = np.abs(tnt_pred - y_te_seq)
    bins     = np.percentile(tnt_std, np.linspace(0, 100, 11))
    bin_idx  = np.digitize(tnt_std, bins[1:-1])
    bin_stds, bin_errs = [], []
    for b in range(10):
        mask = bin_idx == b
        if mask.sum() > 2:
            bin_stds.append(tnt_std[mask].mean())
            bin_errs.append(errors[mask].mean())

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    sc = ax.scatter(bin_stds, bin_errs, c=bin_stds, cmap='Blues',
                    s=60, zorder=3, edgecolors='steelblue', linewidths=0.5)
    plt.colorbar(sc, ax=ax, label='Predicted std')
    lim = max(max(bin_stds), max(bin_errs)) * 1.15
    ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.5, label='Perfect calibration')
    ax.set_xlabel('Predicted std (cycles)')
    ax.set_ylabel('Mean abs error (cycles)')
    ax.set_title('Uncertainty Calibration')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    plt.tight_layout()
    plt.savefig('figures/fig_uncertainty_calibration.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved fig_uncertainty_calibration.png")

    # ── Fig 7: prediction scatter ────────────────────────────────────────────────
    from lstm_baseline import LSTMBaseline, predict_lstm
    lstm = LSTMBaseline(input_dim=17, hidden_dim=64, num_layers=2)
    lstm.load_state_dict(torch.load('lstm_model.pt', map_location='cpu'))
    lstm.eval()
    lstm_pred = predict_lstm(lstm, X_te_seq)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    scatter_pairs = [
        (axes[0], tnt_pred,  y_te_seq, tnt_std,  'Temporal NT (ours)', '#1f77b4'),
        (axes[1], lstm_pred, y_te_seq, None,      'LSTM',               '#2ca02c'),
    ]
    for ax, pred, yt, std, title, col in scatter_pairs:
        ax.scatter(yt, pred, alpha=0.45, s=14, color=col)
        if std is not None:
            ax.errorbar(yt[:25], pred[:25], yerr=std[:25]*1.96,
                        fmt='none', ecolor=col, alpha=0.25, capsize=2)
        lim = 135
        ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.6)
        ax.set_xlim(0, lim); ax.set_ylim(-10, lim)
        ax.set_xlabel('True RUL'); ax.set_ylabel('Predicted RUL')
        ax.set_title(title)
        r = r2(yt, pred)
        ax.text(4, 118, f'RMSE={rmse(yt, pred):.1f}\n$R^2$={r:.3f}',
                fontsize=7.5, color=col)
    plt.tight_layout()
    plt.savefig('figures/fig7_rul_scatter.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved fig7_rul_scatter.png")

except FileNotFoundError as e:
    print(f"Model file not found ({e}) — skipping model-dependent figures.")
    print("Run run_full_experiments.py with model saving enabled first.")

print("\nDone. Copy figures/ to turbofan-paper/figures/ to update paper.")
