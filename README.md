# Robust and Uncertainty-Aware RUL Prediction with Temporal Neural Trees

**Paper:** *Robust and Uncertainty-Aware Remaining Useful Life Prediction with Temporal Neural Trees*

**Author:** Cagri Temel / Hezarfen LLC

**Live Demo:** [Streamlit App](https://share.streamlit.io/cgrtml/turbofan-explainable-neural-trees)

---

## What this is

This project predicts **Remaining Useful Life (RUL)**  the number of flight cycles left before a turbofan engine fails  using a novel model called the **Temporal Neural Tree (TNT)**.

Standard deep-learning baselines (LSTM) are accurate on clean data but degrade severely when sensors fail. TNT combines a GRU sequence encoder with an ensemble of soft decision trees trained with channel-level sensor dropout, making it substantially more robust to missing sensor data while also providing calibrated uncertainty estimates.

---

## Key Results (NASA CMAPSS FD001)

| Model | RMSE (cycles) | R² | Uncertainty | Sensor Dropout |
|---|---|---|---|---|
| **Temporal NT (ours)** | **15.78** | **0.850** | Yes | Yes |
| LSTM | 15.49 | 0.855 | No | No |
| Vanilla NT | 18.64 | 0.794 | No | Yes |
| Random Forest | 18.07 | 0.806 | No | No |

**At 30% missing sensors:** TNT RMSE increases by +2.7 cycles (+17%); LSTM increases by +13.8 cycles (+89%).

---

## Dataset

**NASA CMAPSS** Turbofan Engine Degradation Simulation Dataset  
Source: [NASA Prognostics Center of Excellence](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

This project evaluates on the **FD001 subset**:
- 100 training engines + 100 test engines, single operating condition
- 21 sensor channels per timestep (temperatures, pressures, fan speeds, fuel flow)
- 17 informative features selected after removing constant and near-constant channels
- Target: Remaining Useful Life (RUL) capped at 125 cycles

The FD001 data files are included in `data/` for reproducibility.

---

## Setup

```bash
# 1. Create conda environment (recommended)
conda create -n turbofan python=3.10
conda activate turbofan

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Streamlit demo locally
streamlit run app.py
```

Pre-trained model weights (`tnt_model.pt`, `lstm_model.pt`, `vnt_model.pt`) are included in the repo — the app loads in seconds without retraining.

---

## Project Structure

```
turbofan-explainable-neural-trees/
├── app.py                          # Streamlit interactive demo (6 pages)
├── src/
│   ├── temporal_neural_tree.py     # Temporal Neural Tree: GRU + soft decision trees
│   ├── neural_tree.py              # Vanilla Neural Tree ensemble (baseline)
│   ├── lstm_baseline.py            # LSTM baseline + robustness utilities
│   ├── data_preprocessing.py       # CMAPSS loading, feature engineering, noise/missing utils
│   ├── experiments.py              # Experiment runners
│   └── visualization.py            # Figure generation
├── data/                           # NASA CMAPSS FD001-FD004 .txt files
├── notebooks/
│   └── turbofan_rul_analysis.ipynb # Exploratory notebook
├── results_full.json               # All numerical results (used by fill_paper_numbers.py)
├── tnt_model.pt                    # Pre-trained Temporal Neural Tree weights
├── lstm_model.pt                   # Pre-trained LSTM weights
├── vnt_model.pt                    # Pre-trained Vanilla NT weights
└── requirements.txt
```

---

## Model Architecture

**Temporal Neural Tree (TNT)**

- **GRU Encoder:** 2-layer GRU (hidden=64) encodes the last 30 flight cycles into a fixed-size context vector
- **Soft Decision Tree Ensemble:** 5 trees of depth 5; internal nodes route via sigmoid gating, leaf nodes output Gaussian (mean, variance) pairs
- **Channel-level Sensor Dropout:** during training, entire input channels are zeroed with probability p=0.1, teaching the model to operate under sensor failure
- **Uncertainty:** NLL loss trains leaf variances; final uncertainty is propagated across trees and routing paths — no Monte Carlo sampling needed
- **Feature Importance:** gradient of prediction with respect to each input channel, averaged across timesteps

---

## Experiments

| # | Experiment | Key Finding |
|---|-----------|-------------|
| 1 | Baseline Comparison | TNT within 0.3 RMSE of LSTM on clean data |
| 2 | Missing Sensor Sweep | TNT +17% vs LSTM +89% degradation at 30% missing |
| 3 | Gaussian Noise Sweep | LSTM has slight edge due to temporal averaging; both stay within 3 cycles at σ ≤ 0.05 |
| 4 | Uncertainty Calibration | Pearson r = 0.38 between predicted σ and actual error |

---

## Citation

```bibtex
@article{temel2026temporal,
  title={Robust and Uncertainty-Aware Remaining Useful Life Prediction
         with Temporal Neural Trees},
  author={Temel, Cagri},
  year={2026}
}
```
