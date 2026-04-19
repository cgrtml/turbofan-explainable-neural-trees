"""
Visualization utilities for paper figures.
All figures saved to results/figures/ directory.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

PALETTE = {"Neural Tree": "#2196F3", "Random Forest": "#4CAF50", "XGBoost": "#FF9800"}


def _save(fig, name: str):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_rul_comparison(y_test, results: dict, max_samples: int = 100):
    """Fig 1: Predicted vs True RUL for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Predicted vs True RUL — Model Comparison", fontsize=13, fontweight="bold")

    models = ["Neural Tree", "Random Forest", "XGBoost"]
    idx = np.arange(min(max_samples, len(y_test)))

    for ax, name in zip(axes, models):
        preds = results[name]["predictions"][:max_samples]
        ax.plot(idx, y_test[:max_samples], label="True RUL", color="black",
                linewidth=1.5, linestyle="--")
        ax.plot(idx, preds, label=f"Predicted", color=PALETTE[name],
                linewidth=1.2, alpha=0.85)
        rmse = results[name]["RMSE"]
        mae = results[name]["MAE"]
        r2 = results[name]["R2"]
        ax.set_title(f"{name}\nRMSE={rmse} | MAE={mae} | R²={r2}")
        ax.set_xlabel("Test Sample")
        ax.set_ylabel("RUL (cycles)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "fig1_rul_comparison.png")


def plot_metrics_bar(results: dict):
    """Fig 2: Bar chart comparing RMSE, MAE, R² across models."""
    models = ["Neural Tree", "Random Forest", "XGBoost"]
    metrics = ["RMSE", "MAE", "R2"]
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(12, 4))
    fig.suptitle("Performance Metrics Comparison", fontsize=13, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        values = [results[m][metric] for m in models]
        colors = [PALETTE[m] for m in models]
        bars = ax.bar(models, values, color=colors, alpha=0.85, edgecolor="black",
                      linewidth=0.7)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(values) * 1.2)

    plt.tight_layout()
    _save(fig, "fig2_metrics_bar.png")


def plot_noise_robustness(noise_df: pd.DataFrame):
    """Fig 3: RMSE vs noise level for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Robustness Under Sensor Noise (Gaussian σ)", fontsize=13,
                 fontweight="bold")

    metric_pairs = [("NT_RMSE", "RF_RMSE", "XGB_RMSE", "RMSE"),
                    ("NT_MAE", "RF_MAE", "XGB_MAE", "MAE"),
                    ("NT_R2", "RF_R2", "XGB_R2", "R²")]

    for ax, (nt_col, rf_col, xgb_col, label) in zip(axes, metric_pairs):
        x = noise_df["noise_std"]
        ax.plot(x, noise_df[nt_col], "o-", color=PALETTE["Neural Tree"],
                label="Neural Tree", linewidth=2)
        ax.plot(x, noise_df[rf_col], "s--", color=PALETTE["Random Forest"],
                label="Random Forest", linewidth=2)
        ax.plot(x, noise_df[xgb_col], "^:", color=PALETTE["XGBoost"],
                label="XGBoost", linewidth=2)
        ax.set_xlabel("Noise σ")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs Noise Level")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "fig3_noise_robustness.png")


def plot_missing_sensor_robustness(missing_df: pd.DataFrame):
    """Fig 4: RMSE vs missing sensor ratio for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Robustness Under Missing Sensor Channels", fontsize=13,
                 fontweight="bold")

    metric_pairs = [("NT_RMSE", "RF_RMSE", "XGB_RMSE", "RMSE"),
                    ("NT_MAE", "RF_MAE", "XGB_MAE", "MAE"),
                    ("NT_R2", "RF_R2", "XGB_R2", "R²")]

    for ax, (nt_col, rf_col, xgb_col, label) in zip(axes, metric_pairs):
        x = missing_df["missing_ratio"] * 100
        ax.plot(x, missing_df[nt_col], "o-", color=PALETTE["Neural Tree"],
                label="Neural Tree", linewidth=2)
        ax.plot(x, missing_df[rf_col], "s--", color=PALETTE["Random Forest"],
                label="Random Forest", linewidth=2)
        ax.plot(x, missing_df[xgb_col], "^:", color=PALETTE["XGBoost"],
                label="XGBoost", linewidth=2)
        ax.set_xlabel("Missing Sensors (%)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs Missing Sensor Ratio")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "fig4_missing_sensor_robustness.png")


def plot_feature_importance(importance: np.ndarray, feature_names: list,
                            top_n: int = 10):
    """Fig 5: Neural Tree feature importance (top N sensors)."""
    idx = np.argsort(importance)[::-1][:top_n]
    top_names = [feature_names[i] for i in idx]
    top_vals = importance[idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(top_names[::-1], top_vals[::-1],
                   color=PALETTE["Neural Tree"], alpha=0.85, edgecolor="black",
                   linewidth=0.7)
    ax.set_xlabel("Importance Score (normalized gradient)")
    ax.set_title("Neural Tree — Top Feature Importance", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig5_feature_importance.png")


def plot_training_loss(history: list):
    """Fig 6: Neural Tree training loss curve."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history, color=PALETTE["Neural Tree"], linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Neural Tree Ensemble — Training Loss", fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "fig6_training_loss.png")


def plot_rul_scatter(y_test, results: dict):
    """Fig 7: Scatter plots (True vs Predicted RUL)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("True RUL vs Predicted RUL", fontsize=13, fontweight="bold")

    for ax, name in zip(axes, ["Neural Tree", "Random Forest", "XGBoost"]):
        preds = results[name]["predictions"]
        ax.scatter(y_test, preds, alpha=0.4, s=8, color=PALETTE[name])
        lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
        ax.plot(lims, lims, "k--", linewidth=1.2, label="Ideal")
        ax.set_xlabel("True RUL")
        ax.set_ylabel("Predicted RUL")
        r2 = results[name]["R2"]
        ax.set_title(f"{name} (R²={r2})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, "fig7_rul_scatter.png")
