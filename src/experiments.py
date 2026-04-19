"""
Three core experiments for the paper.

Exp 1: Baseline RUL prediction — Neural Tree vs RF vs XGBoost
Exp 2: Sensor noise robustness
Exp 3: Missing sensor robustness
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from .neural_tree import (
    NeuralTreeEnsemble, train_neural_tree, predict_neural_tree
)
from .data_preprocessing import add_gaussian_noise, simulate_missing_sensors


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": round(rmse, 3), "MAE": round(mae, 3), "R2": round(r2, 4)}


# ─── Experiment 1: Baseline Comparison ────────────────────────────────────────

def run_experiment1(X_train, y_train, X_test, y_test, input_dim: int,
                    epochs: int = 150, verbose: bool = True):
    """RUL prediction: Neural Tree Ensemble vs RF vs XGBoost."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Baseline RUL Prediction Comparison")
    print("="*60)

    results = {}

    # --- Neural Tree Ensemble ---
    print("\n[1/3] Training Neural Tree Ensemble...")
    nt_model = NeuralTreeEnsemble(input_dim=input_dim, n_trees=3, depth=5, dropout=0.1)
    history = train_neural_tree(nt_model, X_train, y_train,
                                epochs=epochs, lr=1e-3, verbose=verbose)
    y_pred_nt = predict_neural_tree(nt_model, X_test)
    results["Neural Tree"] = compute_metrics(y_test, y_pred_nt)
    results["Neural Tree"]["model"] = nt_model
    results["Neural Tree"]["history"] = history
    results["Neural Tree"]["predictions"] = y_pred_nt
    print(f"  → {results['Neural Tree']}")

    # --- Random Forest ---
    print("\n[2/3] Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=12,
                                     random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    results["Random Forest"] = compute_metrics(y_test, y_pred_rf)
    results["Random Forest"]["model"] = rf_model
    results["Random Forest"]["predictions"] = y_pred_rf
    print(f"  → {results['Random Forest']}")

    # --- XGBoost ---
    print("\n[3/3] Training XGBoost...")
    xgb_model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=42, n_jobs=-1, verbosity=0)
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
    y_pred_xgb = xgb_model.predict(X_test)
    results["XGBoost"] = compute_metrics(y_test, y_pred_xgb)
    results["XGBoost"]["model"] = xgb_model
    results["XGBoost"]["predictions"] = y_pred_xgb
    print(f"  → {results['XGBoost']}")

    return results


# ─── Experiment 2: Sensor Noise Robustness ────────────────────────────────────

def run_experiment2(X_train, y_train, X_test, y_test, input_dim: int,
                    noise_levels=None, epochs: int = 150, verbose: bool = False):
    """Evaluate RUL degradation under increasing Gaussian sensor noise."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Sensor Noise Robustness")
    print("="*60)

    if noise_levels is None:
        noise_levels = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]

    # Train once on clean data
    nt_model = NeuralTreeEnsemble(input_dim=input_dim, n_trees=3, depth=5, dropout=0.1)
    train_neural_tree(nt_model, X_train, y_train, epochs=epochs, verbose=verbose)

    rf_model = RandomForestRegressor(n_estimators=200, max_depth=12,
                                     random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    xgb_model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                              random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train)

    noise_results = []

    for std in noise_levels:
        X_noisy = add_gaussian_noise(X_test, noise_std=std) if std > 0 else X_test.copy()

        row = {"noise_std": std}

        y_pred_nt = predict_neural_tree(nt_model, X_noisy)
        m_nt = compute_metrics(y_test, y_pred_nt)
        row["NT_RMSE"] = m_nt["RMSE"]
        row["NT_MAE"] = m_nt["MAE"]
        row["NT_R2"] = m_nt["R2"]

        y_pred_rf = rf_model.predict(X_noisy)
        m_rf = compute_metrics(y_test, y_pred_rf)
        row["RF_RMSE"] = m_rf["RMSE"]
        row["RF_MAE"] = m_rf["MAE"]
        row["RF_R2"] = m_rf["R2"]

        y_pred_xgb = xgb_model.predict(X_noisy)
        m_xgb = compute_metrics(y_test, y_pred_xgb)
        row["XGB_RMSE"] = m_xgb["RMSE"]
        row["XGB_MAE"] = m_xgb["MAE"]
        row["XGB_R2"] = m_xgb["R2"]

        noise_results.append(row)
        print(f"  noise={std:.2f} | NT RMSE={row['NT_RMSE']:.3f} | "
              f"RF RMSE={row['RF_RMSE']:.3f} | XGB RMSE={row['XGB_RMSE']:.3f}")

    return pd.DataFrame(noise_results)


# ─── Experiment 3: Missing Sensor Robustness ──────────────────────────────────

def run_experiment3(X_train, y_train, X_test, y_test, input_dim: int,
                    missing_ratios=None, epochs: int = 150, verbose: bool = False):
    """Evaluate RUL degradation under increasing sensor dropout."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Missing Sensor Scenario")
    print("="*60)

    if missing_ratios is None:
        missing_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    nt_model = NeuralTreeEnsemble(input_dim=input_dim, n_trees=3, depth=5, dropout=0.1)
    train_neural_tree(nt_model, X_train, y_train, epochs=epochs, verbose=verbose)

    rf_model = RandomForestRegressor(n_estimators=200, max_depth=12,
                                     random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    xgb_model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                              random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train)

    missing_results = []

    for ratio in missing_ratios:
        if ratio > 0:
            X_missing, dropped_idx = simulate_missing_sensors(X_test, missing_ratio=ratio)
        else:
            X_missing = X_test.copy()
            dropped_idx = []

        row = {"missing_ratio": ratio, "n_dropped": len(dropped_idx)}

        y_pred_nt = predict_neural_tree(nt_model, X_missing)
        m_nt = compute_metrics(y_test, y_pred_nt)
        row["NT_RMSE"] = m_nt["RMSE"]
        row["NT_MAE"] = m_nt["MAE"]
        row["NT_R2"] = m_nt["R2"]

        y_pred_rf = rf_model.predict(X_missing)
        m_rf = compute_metrics(y_test, y_pred_rf)
        row["RF_RMSE"] = m_rf["RMSE"]
        row["RF_MAE"] = m_rf["MAE"]
        row["RF_R2"] = m_rf["R2"]

        y_pred_xgb = xgb_model.predict(X_missing)
        m_xgb = compute_metrics(y_test, y_pred_xgb)
        row["XGB_RMSE"] = m_xgb["RMSE"]
        row["XGB_MAE"] = m_xgb["MAE"]
        row["XGB_R2"] = m_xgb["R2"]

        missing_results.append(row)
        print(f"  missing={ratio:.1f} | NT RMSE={row['NT_RMSE']:.3f} | "
              f"RF RMSE={row['RF_RMSE']:.3f} | XGB RMSE={row['XGB_RMSE']:.3f}")

    return pd.DataFrame(missing_results)
