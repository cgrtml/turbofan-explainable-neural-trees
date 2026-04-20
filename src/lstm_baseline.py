"""
LSTM Baseline for CMAPSS RUL prediction.
Trained on full time-series sequences (no sensor dropout).
Used to demonstrate robustness gap vs Neural Tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim: int = 17, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc   = nn.Linear(hidden_dim, 1)
        self._y_min = 0.0
        self._y_max = 130.0

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])
        return self.fc(out).squeeze(-1)


def train_lstm(model: LSTMBaseline, X_train: np.ndarray, y_train: np.ndarray,
               epochs: int = 150, lr: float = 1e-3, batch_size: int = 512,
               verbose: bool = True) -> list:
    model._y_min = float(y_train.min())
    model._y_max = float(y_train.max())
    y_norm = (y_train - model._y_min) / (model._y_max - model._y_min + 1e-8)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_norm,  dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t),
        batch_size=batch_size, shuffle=True
    )

    history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg = epoch_loss / len(loader)
        history.append(avg)
        if verbose and (epoch + 1) % 30 == 0:
            print(f"  LSTM Epoch {epoch+1:3d}/{epochs} | Loss: {avg:.6f}")

    return history


def predict_lstm(model: LSTMBaseline, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return pred * (model._y_max - model._y_min) + model._y_min


def apply_missing_to_sequences(X_seq: np.ndarray, missing_ratio: float,
                                seed: int = 42):
    """Zero out entire sensor channels across all timesteps in a sequence batch."""
    rng   = np.random.default_rng(seed)
    X_m   = X_seq.copy()
    n_feat = X_seq.shape[2]
    n_drop = int(n_feat * missing_ratio)
    if n_drop == 0:
        return X_m, np.array([], dtype=int)
    idx = rng.choice(n_feat, n_drop, replace=False)
    X_m[:, :, idx] = 0.0
    return X_m, idx


def apply_noise_to_sequences(X_seq: np.ndarray, noise_std: float,
                              seed: int = 42) -> np.ndarray:
    rng   = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std, X_seq.shape).astype(np.float32)
    return np.clip(X_seq + noise, 0.0, 1.0)
