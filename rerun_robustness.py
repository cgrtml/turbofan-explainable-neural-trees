"""
Re-runs missing sensor + noise sweeps with corrected apply_missing_to_sequences
(removed max(1,...) bug). Also computes real Pearson r for calibration.
Loads saved models — no retraining needed.
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import json, torch
from math import sqrt
from scipy.stats import pearsonr

from data_preprocessing import (load_cmapss, prepare_features,
                                 load_cmapss_sequences,
                                 add_gaussian_noise, simulate_missing_sensors)
from temporal_neural_tree import TemporalNeuralTreeEnsemble, predict_temporal_nt
from neural_tree import NeuralTreeEnsemble, predict_neural_tree
from lstm_baseline import LSTMBaseline, predict_lstm, apply_missing_to_sequences, apply_noise_to_sequences
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

torch.manual_seed(42)
np.random.seed(42)

def rmse(yt, yp): return float(sqrt(np.mean((np.array(yt)-np.array(yp))**2)))

print("Loading data...")
train_df, test_df = load_cmapss('FD001', 'data')
X_train, y_train, X_test, y_test, _ = prepare_features(train_df, test_df)
X_tr_seq, y_tr_seq, X_te_seq, y_te_seq, _ = load_cmapss_sequences('FD001', 'data', seq_len=30)

print("Loading saved models...")
tnt = TemporalNeuralTreeEnsemble(input_dim=17, hidden_dim=64, gru_layers=2, n_trees=5, depth=5)
tnt.load_state_dict(torch.load('tnt_model.pt', map_location='cpu'))
tnt.eval()

from neural_tree import train_neural_tree
print("Training VanillaNT (fast, ~2 min)...")
vnt = NeuralTreeEnsemble(input_dim=17, n_trees=5, depth=5, hidden_dim=32)
train_neural_tree(vnt, X_train, y_train, epochs=200, verbose=False, sensor_dropout=0.1)
torch.save(vnt.state_dict(), 'vnt_model.pt')
vnt.eval()

lstm = LSTMBaseline(input_dim=17, hidden_dim=64, num_layers=2)
lstm.load_state_dict(torch.load('lstm_model.pt', map_location='cpu'))
lstm.eval()

# Re-train RF/GB (fast, no save needed)
print("Training RF & GB (fast)...")
rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
gb = None  # not used in robustness tables

# ── baseline predictions ──────────────────────────────────────────────────────
tnt_pred, tnt_std = predict_temporal_nt(tnt, X_te_seq)
vnt_pred          = predict_neural_tree(vnt, X_test)
lstm_pred         = predict_lstm(lstm, X_te_seq)
rf_pred           = rf.predict(X_test)

print(f"Baseline check: TNT={rmse(y_te_seq,tnt_pred):.3f} LSTM={rmse(y_te_seq,lstm_pred):.3f}")

# ── missing sensor sweep ──────────────────────────────────────────────────────
print("\nMissing sensor sweep (with fix)...")
miss_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
miss_res = {n: [] for n in ['TemporalNT','VanillaNT','LSTM','RF']}

for r in miss_ratios:
    Xm_snap, _ = simulate_missing_sensors(X_test, r)
    Xm_seq,  _ = apply_missing_to_sequences(X_te_seq, r)

    miss_res['TemporalNT'].append(rmse(y_te_seq, predict_temporal_nt(tnt, Xm_seq)[0]))
    miss_res['VanillaNT'].append(rmse(y_test,   predict_neural_tree(vnt, Xm_snap)))
    miss_res['LSTM'].append(     rmse(y_te_seq, predict_lstm(lstm, Xm_seq)))
    miss_res['RF'].append(       rmse(y_test,   rf.predict(Xm_snap)))

    line = f"  r={r:.0%}"
    for n in ['TemporalNT','VanillaNT','LSTM','RF']:
        line += f" | {n}={miss_res[n][-1]:.2f}"
    print(line)

# ── noise sweep ───────────────────────────────────────────────────────────────
print("\nNoise sweep...")
sigmas = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]
noise_res = {n: [] for n in ['TemporalNT','VanillaNT','LSTM','RF']}

for s in sigmas:
    Xn_snap = add_gaussian_noise(X_test,   s) if s > 0 else X_test.copy()
    Xn_seq  = apply_noise_to_sequences(X_te_seq, s) if s > 0 else X_te_seq.copy()

    noise_res['TemporalNT'].append(rmse(y_te_seq, predict_temporal_nt(tnt, Xn_seq)[0]))
    noise_res['VanillaNT'].append(rmse(y_test,   predict_neural_tree(vnt, Xn_snap)))
    noise_res['LSTM'].append(     rmse(y_te_seq, predict_lstm(lstm, Xn_seq)))
    noise_res['RF'].append(       rmse(y_test,   rf.predict(Xn_snap)))

# ── calibration Pearson r ─────────────────────────────────────────────────────
errors = np.abs(tnt_pred - y_te_seq)
bins   = np.percentile(tnt_std, np.linspace(0, 100, 11))
bin_idx = np.digitize(tnt_std, bins[1:-1])
bin_stds, bin_errs = [], []
for b in range(10):
    mask = bin_idx == b
    if mask.sum() > 2:
        bin_stds.append(tnt_std[mask].mean())
        bin_errs.append(errors[mask].mean())

r_val, p_val = pearsonr(bin_stds, bin_errs)
print(f"\nCalibration Pearson r = {r_val:.3f} (p={p_val:.4f})")

# ── update results_full.json ──────────────────────────────────────────────────
with open('results_full.json') as f:
    res = json.load(f)

def to_python(obj):
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32, np.int64)): return int(obj)
    if isinstance(obj, list): return [to_python(x) for x in obj]
    if isinstance(obj, dict): return {k: to_python(v) for k, v in obj.items()}
    return obj

res['missing_sensor'] = to_python({'ratios': miss_ratios, **miss_res})
res['noise']          = to_python({'sigmas': sigmas, **noise_res})
res['calibration_r']  = round(float(r_val), 3)

with open('results_full.json', 'w') as f:
    json.dump(res, f, indent=2)

print("\nresults_full.json updated.")
print("\nCopy these to main.tex Table III:")
for i, r in enumerate(miss_ratios):
    print(f"  {int(r*100)}%: TNT={miss_res['TemporalNT'][i]:.2f} "
          f"LSTM={miss_res['LSTM'][i]:.2f} "
          f"VNT={miss_res['VanillaNT'][i]:.2f} "
          f"RF={miss_res['RF'][i]:.2f}")
print(f"\nCalibration r = {r_val:.3f} → use in paper")
