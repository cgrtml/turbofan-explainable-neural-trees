"""
Full experiment suite: TemporalNT vs VanillaNT vs LSTM vs RF vs GB
Generates all results needed for the upgraded paper.
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import sqrt
import json, os

import torch
torch.manual_seed(42)
np.random.seed(42)

from data_preprocessing import (load_cmapss, prepare_features,
                                 load_cmapss_sequences,
                                 add_gaussian_noise, simulate_missing_sensors)
from neural_tree import NeuralTreeEnsemble, train_neural_tree, predict_neural_tree
from temporal_neural_tree import (TemporalNeuralTreeEnsemble, train_temporal_nt,
                                   predict_temporal_nt)
from lstm_baseline import (LSTMBaseline, train_lstm, predict_lstm,
                            apply_missing_to_sequences, apply_noise_to_sequences)
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

os.makedirs('figures', exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def rmse(yt, yp): return float(sqrt(np.mean((yt - yp) ** 2)))
def mae(yt, yp):  return float(np.mean(np.abs(yt - yp)))
def r2(yt, yp):
    return 1 - np.sum((yt - yp)**2) / np.sum((yt - np.mean(yt))**2)
def nasa(yt, yp):
    d = yp - yt
    return float(np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1).sum())

def metrics(yt, yp):
    return dict(RMSE=round(rmse(yt,yp),3), MAE=round(mae(yt,yp),3),
                R2=round(r2(yt,yp),4),   NASA=round(nasa(yt,yp),1))

# ── data ─────────────────────────────────────────────────────────────────────
print("Loading data...")
train_df, test_df = load_cmapss('FD001', 'data')
X_train, y_train, X_test, y_test, _ = prepare_features(train_df, test_df)
X_tr_seq, y_tr_seq, X_te_seq, y_te_seq, _ = \
    load_cmapss_sequences('FD001', 'data', seq_len=30)

# ── train models ─────────────────────────────────────────────────────────────
print("\nTraining VanillaNT (snapshot, sensor dropout p=0.1)...")
vnt = NeuralTreeEnsemble(input_dim=17, n_trees=5, depth=5, hidden_dim=32)
train_neural_tree(vnt, X_train, y_train, epochs=200,
                  verbose=True, sensor_dropout=0.1)

print("\nTraining TemporalNT (30-cycle sequence, sensor dropout p=0.1)...")
tnt = TemporalNeuralTreeEnsemble(input_dim=17, hidden_dim=64,
                                  gru_layers=2, n_trees=5, depth=5)
train_temporal_nt(tnt, X_tr_seq, y_tr_seq, epochs=200,
                  verbose=True, sensor_dropout=0.1)

print("\nTraining LSTM (30-cycle sequence, NO sensor dropout)...")
lstm = LSTMBaseline(input_dim=17, hidden_dim=64, num_layers=2)
train_lstm(lstm, X_tr_seq, y_tr_seq, epochs=150, verbose=True)

print("\nTraining RF & GB...")
rf = RandomForestRegressor(n_estimators=200, max_depth=12,
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
gb = HistGradientBoostingRegressor(max_iter=200, max_depth=5,
                                    learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)

# ── clean predictions ─────────────────────────────────────────────────────────
vnt_pred             = predict_neural_tree(vnt, X_test)
tnt_pred, tnt_std    = predict_temporal_nt(tnt, X_te_seq)
lstm_pred            = predict_lstm(lstm, X_te_seq)
rf_pred              = rf.predict(X_test)
gb_pred              = gb.predict(X_test)

print("\n=== CLEAN-DATA BASELINE ===")
results = {}
for name, pred, yt in [('TemporalNT', tnt_pred, y_te_seq),
                        ('VanillaNT',  vnt_pred, y_test),
                        ('LSTM',       lstm_pred, y_te_seq),
                        ('RF',         rf_pred,  y_test),
                        ('GB',         gb_pred,  y_test)]:
    m = metrics(yt, pred)
    results[name] = m
    print(f"  {name:12s} | RMSE={m['RMSE']:.3f} | MAE={m['MAE']:.3f} "
          f"| R2={m['R2']:.4f} | NASA={m['NASA']:.1f}")

# ── missing sensor robustness ─────────────────────────────────────────────────
print("\n=== MISSING SENSOR ROBUSTNESS ===")
miss_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
miss_res = {n: [] for n in ['TemporalNT','VanillaNT','LSTM','RF','GB']}

for r in miss_ratios:
    Xm_snap, _ = simulate_missing_sensors(X_test, r)
    Xm_seq,  _ = apply_missing_to_sequences(X_te_seq, r)

    tnt_p, _ = predict_temporal_nt(tnt, Xm_seq)
    miss_res['TemporalNT'].append(rmse(y_te_seq, tnt_p))
    miss_res['VanillaNT'].append(rmse(y_test,    predict_neural_tree(vnt, Xm_snap)))
    miss_res['LSTM'].append(     rmse(y_te_seq,  predict_lstm(lstm, Xm_seq)))
    miss_res['RF'].append(       rmse(y_test,    rf.predict(Xm_snap)))
    miss_res['GB'].append(       rmse(y_test,    gb.predict(Xm_snap)))

    line = f"  r={r:.0%}"
    for n in ['TemporalNT','VanillaNT','LSTM','RF','GB']:
        line += f" | {n}={miss_res[n][-1]:.2f}"
    print(line)

# ── noise robustness ──────────────────────────────────────────────────────────
print("\n=== NOISE ROBUSTNESS ===")
sigmas = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]
noise_res = {n: [] for n in ['TemporalNT','VanillaNT','LSTM','RF','GB']}

for s in sigmas:
    Xn_snap = add_gaussian_noise(X_test,   s) if s > 0 else X_test.copy()
    Xn_seq  = apply_noise_to_sequences(X_te_seq, s) if s > 0 else X_te_seq.copy()

    tnt_p, _ = predict_temporal_nt(tnt, Xn_seq)
    noise_res['TemporalNT'].append(rmse(y_te_seq, tnt_p))
    noise_res['VanillaNT'].append(rmse(y_test,    predict_neural_tree(vnt, Xn_snap)))
    noise_res['LSTM'].append(     rmse(y_te_seq,  predict_lstm(lstm, Xn_seq)))
    noise_res['RF'].append(       rmse(y_test,    rf.predict(Xn_snap)))
    noise_res['GB'].append(       rmse(y_test,    gb.predict(Xn_snap)))

    line = f"  s={s:.2f}"
    for n in ['TemporalNT','VanillaNT','LSTM','RF','GB']:
        line += f" | {n}={noise_res[n][-1]:.2f}"
    print(line)

# ── uncertainty calibration ───────────────────────────────────────────────────
print("\n=== UNCERTAINTY CALIBRATION (TemporalNT) ===")
errors = np.abs(tnt_pred - y_te_seq)
# bin by predicted std
bins = np.percentile(tnt_std, np.linspace(0,100,11))
bin_idx = np.digitize(tnt_std, bins[1:-1])
for b in range(10):
    mask = bin_idx == b
    if mask.sum() > 0:
        avg_std = tnt_std[mask].mean()
        avg_err = errors[mask].mean()
        print(f"  std_bin={avg_std:.2f} | mean_abs_err={avg_err:.2f} | n={mask.sum()}")

# ── feature importance ────────────────────────────────────────────────────────
print("\n=== FEATURE IMPORTANCE (TemporalNT) ===")
from data_preprocessing import USEFUL_FEATURES
feat_imp = tnt.get_feature_importance(torch.tensor(X_te_seq, dtype=torch.float32))
feat_names = USEFUL_FEATURES
top_idx = np.argsort(feat_imp)[::-1][:10]
for i in top_idx:
    print(f"  {feat_names[i]:4s}: {feat_imp[i]:.4f}")

# ── figures ───────────────────────────────────────────────────────────────────
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9,
                     'legend.fontsize': 7.5, 'lines.linewidth': 1.5,
                     'lines.markersize': 4.5})

COLORS = {'TemporalNT': 'black', 'VanillaNT': 'dimgray',
          'LSTM': 'gray', 'RF': 'darkgray', 'GB': 'lightgray'}
MARKS  = {'TemporalNT': 'o', 'VanillaNT': 's', 'LSTM': '^',
          'RF': 'D', 'GB': 'v'}
LS     = {'TemporalNT': '-', 'VanillaNT': '--', 'LSTM': '-.',
          'RF': ':', 'GB': ':'}

# Fig: missing sensor
fig, ax = plt.subplots(figsize=(3.5, 2.7))
x_pct = [r*100 for r in miss_ratios]
for n in ['TemporalNT','VanillaNT','LSTM','RF']:
    lbl = 'Temporal NT (ours)' if n == 'TemporalNT' else \
          'Vanilla NT' if n == 'VanillaNT' else n
    ax.plot(x_pct, miss_res[n], color=COLORS[n], marker=MARKS[n],
            ls=LS[n], label=lbl, zorder=3 if 'NT' in n else 2)
ax.set_xlabel('Missing sensor channels (%)')
ax.set_ylabel('RMSE (cycles)')
ax.set_title('Missing Sensor Robustness — FD001')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3, linewidth=0.5)
plt.tight_layout()
plt.savefig('figures/fig4_missing_sensor_robustness.png', dpi=200, bbox_inches='tight')
plt.close()

# Fig: noise
fig, ax = plt.subplots(figsize=(3.5, 2.7))
for n in ['TemporalNT','VanillaNT','LSTM','RF']:
    lbl = 'Temporal NT (ours)' if n == 'TemporalNT' else \
          'Vanilla NT' if n == 'VanillaNT' else n
    ax.plot(sigmas, noise_res[n], color=COLORS[n], marker=MARKS[n],
            ls=LS[n], label=lbl, zorder=3 if 'NT' in n else 2)
ax.set_xlabel('Gaussian noise std $\\sigma$')
ax.set_ylabel('RMSE (cycles)')
ax.set_title('Noise Robustness — FD001')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3, linewidth=0.5)
plt.tight_layout()
plt.savefig('figures/fig3_noise_robustness.png', dpi=200, bbox_inches='tight')
plt.close()

# Fig: uncertainty calibration
fig, ax = plt.subplots(figsize=(3.2, 2.6))
bin_stds, bin_errs = [], []
for b in range(10):
    mask = bin_idx == b
    if mask.sum() > 2:
        bin_stds.append(tnt_std[mask].mean())
        bin_errs.append(errors[mask].mean())
ax.scatter(bin_stds, bin_errs, color='black', s=35, zorder=3)
lim = max(max(bin_stds), max(bin_errs)) * 1.1
ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.5, label='Perfect calibration')
ax.set_xlabel('Predicted std (cycles)')
ax.set_ylabel('Mean abs error (cycles)')
ax.set_title('Uncertainty Calibration — TemporalNT')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, linewidth=0.5)
plt.tight_layout()
plt.savefig('figures/fig_uncertainty_calibration.png', dpi=200, bbox_inches='tight')
plt.close()

# Fig: feature importance
fig, ax = plt.subplots(figsize=(3.5, 2.7))
top_names = [feat_names[i] for i in top_idx]
top_vals  = [feat_imp[i] * 100 for i in top_idx]
ax.barh(range(len(top_names)), top_vals[::-1], color='black', alpha=0.75)
ax.set_yticks(range(len(top_names)))
ax.set_yticklabels(top_names[::-1], fontsize=8)
ax.set_xlabel('Importance (%)')
ax.set_title('Top-10 Feature Importance — TemporalNT')
ax.grid(True, alpha=0.3, linewidth=0.5, axis='x')
plt.tight_layout()
plt.savefig('figures/fig5_feature_importance.png', dpi=200, bbox_inches='tight')
plt.close()

# Fig: prediction scatter (TemporalNT vs LSTM)
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
for ax, pred, yt, title in [
    (axes[0], tnt_pred,  y_te_seq, 'Temporal NT (ours)'),
    (axes[1], lstm_pred, y_te_seq, 'LSTM')]:
    ax.scatter(yt, pred, alpha=0.5, s=14, color='black')
    ax.errorbar(yt[:20], tnt_pred[:20] if 'NT' in title else lstm_pred[:20],
                yerr=tnt_std[:20] if 'NT' in title else None,
                fmt='none', ecolor='gray', alpha=0.3, capsize=2)
    lim = 135
    ax.plot([0, lim], [0, lim], 'k--', lw=0.8)
    ax.set_xlim(0, lim); ax.set_ylim(-10, lim)
    ax.set_xlabel('True RUL'); ax.set_ylabel('Predicted RUL')
    ax.set_title(title)
    r = r2(yt, pred)
    ax.text(5, 120, f'RMSE={rmse(yt,pred):.1f}\n$R^2$={r:.3f}', fontsize=7.5)
plt.tight_layout()
plt.savefig('figures/fig7_rul_scatter.png', dpi=200, bbox_inches='tight')
plt.close()

print("\nAll figures saved.")

# ── save all results to JSON for paper ───────────────────────────────────────
output = {
    'baseline': results,
    'missing_sensor': {
        'ratios': miss_ratios,
        **miss_res
    },
    'noise': {
        'sigmas': sigmas,
        **noise_res
    }
}
with open('results_full.json', 'w') as f:
    json.dump(output, f, indent=2)
print("Results saved to results_full.json")
