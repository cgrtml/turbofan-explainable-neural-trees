# Explainable Neural Trees for RUL Prediction in Turbofan Engines

**Paper:** *Explainable Neural Trees for Remaining Useful Life Prediction in Turbofan Engines Under Sensor Noise and Missing Data*

**Submitted to:** IEEE SMC 2026 Workshop on Trustworthy AI for Safety-Critical Perception and Decision Systems

---

## Dataset

**NASA CMAPSS** — Turbofan Engine Degradation Simulation Dataset  
Source: [NASA Prognostics Center of Excellence](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

- 4 sub-datasets (FD001–FD004), 100–260 training engines per subset
- 21 sensor channels + 3 operational settings per time step
- Target: Remaining Useful Life (RUL) in flight cycles

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NASA CMAPSS data
python download_data.py

# 4. Launch notebook
jupyter notebook notebooks/turbofan_rul_analysis.ipynb
```

## Experiments

| # | Experiment | Description |
|---|-----------|-------------|
| 1 | Baseline Comparison | Neural Tree vs Random Forest vs XGBoost — RMSE, MAE, R² |
| 2 | Sensor Noise Robustness | Gaussian noise σ ∈ {0, 0.02, 0.05, 0.10, 0.15, 0.20} applied at inference |
| 3 | Missing Sensor Scenario | Random sensor dropout 0%–50% simulating sensor failure |

## Project Structure

```
turbofan-explainable-neural-trees/
├── src/
│   ├── neural_tree.py          # Soft Neural Decision Tree (PyTorch)
│   ├── data_preprocessing.py   # CMAPSS loading & feature engineering
│   ├── experiments.py          # All 3 experiments
│   └── visualization.py        # Paper figures
├── notebooks/
│   └── turbofan_rul_analysis.ipynb   # Main notebook
├── data/                       # CMAPSS .txt files (download separately)
├── results/
│   ├── figures/                # Generated paper figures
│   ├── table1_baseline.csv
│   ├── table2_noise_robustness.csv
│   └── table3_missing_sensor.csv
├── download_data.py            # Dataset downloader
└── requirements.txt
```

## Model Architecture

**Soft Neural Decision Tree (NeuralTreeEnsemble)**
- Ensemble of 3 soft decision trees, each with depth=5
- Internal nodes: linear routing with sigmoid activation + batch normalization
- Leaf nodes: learned scalar RUL estimates
- Training: Adam optimizer + cosine annealing, MSE loss + L2 regularization
- Explainability: input gradient-based feature importance

## Citation

If you use this code in your research, please cite:

```
@inproceedings{temel2026explainable,
  title={Explainable Neural Trees for Remaining Useful Life Prediction
         in Turbofan Engines Under Sensor Noise and Missing Data},
  author={Temel, Cagri},
  booktitle={Proceedings of IEEE SMC 2026 Workshop on Trustworthy AI
             for Safety-Critical Perception and Decision Systems},
  year={2026}
}
```
