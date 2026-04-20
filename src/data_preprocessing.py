"""
CMAPSS Turbofan Engine Degradation Dataset - Data Preprocessing
NASA Prognostics Center of Excellence
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import urllib.request
import zipfile

SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
OP_COLS = ["op1", "op2", "op3"]
FEATURE_COLS = OP_COLS + SENSOR_COLS

# Sensors with near-zero variance (not informative) — commonly dropped
DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
USEFUL_FEATURES = [c for c in FEATURE_COLS if c not in DROP_SENSORS]

MAX_RUL = 130  # RUL cap for piece-wise linear target


def download_cmapss(data_dir: str = "data"):
    """Download CMAPSS dataset from NASA if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    fd001_train = os.path.join(data_dir, "train_FD001.txt")
    if os.path.exists(fd001_train):
        print("CMAPSS data already present.")
        return

    url = "https://ti.arc.nasa.gov/c/6/"
    zip_path = os.path.join(data_dir, "cmapss.zip")
    print(f"Downloading CMAPSS dataset...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
        print("Download complete.")
    except Exception as e:
        print(f"Auto-download failed: {e}")
        print("Please manually download from:")
        print("https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
        print(f"and place .txt files in '{data_dir}/' directory.")


def load_cmapss(dataset: str = "FD001", data_dir: str = "data"):
    """Load and preprocess CMAPSS dataset."""
    col_names = ["unit_id", "time_cycle"] + OP_COLS + SENSOR_COLS

    train_path = os.path.join(data_dir, f"train_{dataset}.txt")
    test_path = os.path.join(data_dir, f"test_{dataset}.txt")
    rul_path = os.path.join(data_dir, f"RUL_{dataset}.txt")

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=col_names)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=col_names)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])

    # Compute RUL for training set (piece-wise linear)
    train_df = _compute_rul(train_df)

    # Compute RUL for test set (use last cycle + provided RUL)
    test_df = _compute_test_rul(test_df, rul_df)

    return train_df, test_df


def _compute_rul(df: pd.DataFrame) -> pd.DataFrame:
    max_cycles = df.groupby("unit_id")["time_cycle"].max().reset_index()
    max_cycles.columns = ["unit_id", "max_cycle"]
    df = df.merge(max_cycles, on="unit_id")
    df["RUL"] = df["max_cycle"] - df["time_cycle"]
    df["RUL"] = df["RUL"].clip(upper=MAX_RUL)
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def _compute_test_rul(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    # Use last recorded cycle per unit
    last_cycles = test_df.groupby("unit_id").last().reset_index()
    last_cycles["RUL"] = rul_df["RUL"].values
    last_cycles["RUL"] = last_cycles["RUL"].clip(upper=MAX_RUL)
    return last_cycles


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame,
                     feature_cols: list = None, scale: bool = True):
    """Extract and scale features."""
    if feature_cols is None:
        feature_cols = USEFUL_FEATURES

    X_train = train_df[feature_cols].values
    y_train = train_df["RUL"].values.astype(np.float32)

    X_test = test_df[feature_cols].values
    y_test = test_df["RUL"].values.astype(np.float32)

    if scale:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        scaler = None

    return X_train, y_train, X_test, y_test, scaler


def load_cmapss_sequences(dataset: str = 'FD001', data_dir: str = 'data',
                          seq_len: int = 30):
    """Load CMAPSS as fixed-length sequences for LSTM training.

    Returns:
        X_train_seq: (N_train, seq_len, n_features)
        y_train:     (N_train,)
        X_test_seq:  (N_test,  seq_len, n_features)  — one per engine
        y_test:      (N_test,)
        scaler:      fitted MinMaxScaler
    """
    col_names = ["unit_id", "time_cycle"] + OP_COLS + SENSOR_COLS

    train_df  = pd.read_csv(os.path.join(data_dir, f"train_{dataset}.txt"),
                            sep=r"\s+", header=None, names=col_names)
    test_df   = pd.read_csv(os.path.join(data_dir, f"test_{dataset}.txt"),
                            sep=r"\s+", header=None, names=col_names)
    rul_df    = pd.read_csv(os.path.join(data_dir, f"RUL_{dataset}.txt"),
                            sep=r"\s+", header=None, names=["RUL"])

    train_df  = _compute_rul(train_df)
    test_full = _compute_test_rul_full(test_df, rul_df)

    scaler = MinMaxScaler()
    scaler.fit(train_df[USEFUL_FEATURES].values)

    X_tr, y_tr = _make_sequences(train_df, scaler, seq_len)
    X_te, y_te = _make_test_sequences(test_full, scaler, seq_len)

    return X_tr, y_tr, X_te, y_te, scaler


def _compute_test_rul_full(test_df: pd.DataFrame,
                            rul_df: pd.DataFrame) -> pd.DataFrame:
    """Assign a RUL value to every row in the raw test file."""
    parts = []
    for idx, (uid, grp) in enumerate(test_df.groupby("unit_id", sort=True)):
        grp = grp.copy().sort_values("time_cycle").reset_index(drop=True)
        true_rul = float(rul_df.iloc[idx]["RUL"])
        max_t    = grp["time_cycle"].max()
        grp["RUL"] = (max_t - grp["time_cycle"]) + true_rul
        grp["RUL"] = grp["RUL"].clip(upper=MAX_RUL)
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)


def _make_sequences(df: pd.DataFrame, scaler, seq_len: int):
    seqs, tgts = [], []
    for uid in df["unit_id"].unique():
        eng  = df[df["unit_id"] == uid].sort_values("time_cycle")
        feat = scaler.transform(eng[USEFUL_FEATURES].values).astype(np.float32)
        ruls = eng["RUL"].values.astype(np.float32)
        for i in range(len(feat)):
            s = max(0, i - seq_len + 1)
            seq = feat[s: i + 1]
            if len(seq) < seq_len:
                pad = np.zeros((seq_len - len(seq), feat.shape[1]), dtype=np.float32)
                seq = np.vstack([pad, seq])
            seqs.append(seq)
            tgts.append(ruls[i])
    return np.array(seqs, dtype=np.float32), np.array(tgts, dtype=np.float32)


def _make_test_sequences(df: pd.DataFrame, scaler, seq_len: int):
    seqs, tgts = [], []
    for uid in df["unit_id"].unique():
        eng  = df[df["unit_id"] == uid].sort_values("time_cycle")
        feat = scaler.transform(eng[USEFUL_FEATURES].values).astype(np.float32)
        seq  = feat[-seq_len:]
        if len(seq) < seq_len:
            pad = np.zeros((seq_len - len(seq), feat.shape[1]), dtype=np.float32)
            seq = np.vstack([pad, seq])
        seqs.append(seq)
        tgts.append(float(eng["RUL"].values[-1]))
    return np.array(seqs, dtype=np.float32), np.array(tgts, dtype=np.float32)


def add_gaussian_noise(X: np.ndarray, noise_std: float = 0.05,
                       seed: int = 42) -> np.ndarray:
    """Add Gaussian noise to simulate sensor degradation."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std, X.shape)
    return np.clip(X + noise, 0, 1)


def simulate_missing_sensors(X: np.ndarray, missing_ratio: float = 0.2,
                              seed: int = 42) -> np.ndarray:
    """Zero-out random sensors to simulate sensor failure."""
    rng = np.random.default_rng(seed)
    X_missing = X.copy()
    n_features = X.shape[1]
    n_missing = int(n_features * missing_ratio)
    if n_missing == 0:
        return X_missing, np.array([], dtype=int)
    missing_idx = rng.choice(n_features, n_missing, replace=False)
    X_missing[:, missing_idx] = 0.0
    return X_missing, missing_idx
