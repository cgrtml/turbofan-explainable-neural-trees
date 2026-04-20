"""
Temporal Neural Tree Ensemble with Uncertainty Quantification.

Architecture:
  Input (batch, seq_len, n_feat)
    → GRU Encoder (2 layers) → last hidden state (batch, hidden_dim)
    → 5 Soft Decision Trees, each outputting (mean, log_var) per leaf
    → Ensemble mean + epistemic uncertainty

Key improvements over vanilla NeuralTreeEnsemble:
  - Processes full 30-cycle sequences (same as LSTM) → closes accuracy gap
  - Sensor dropout at channel level during training → missing-sensor robustness
  - Leaf-level Gaussian uncertainty → calibrated confidence intervals
  - Structural interpretability via gradient-based feature importance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UncertainSoftDecisionTree(nn.Module):
    """Soft decision tree with per-leaf Gaussian output (mean + log variance)."""

    def __init__(self, input_dim: int, depth: int = 5):
        super().__init__()
        self.depth = depth
        n_internal = 2 ** depth - 1
        n_leaves = 2 ** depth

        self.node_weights = nn.Parameter(torch.randn(n_internal, input_dim) * 0.01)
        self.node_biases = nn.Parameter(torch.zeros(n_internal))
        self.leaf_means = nn.Parameter(torch.linspace(0, 1, n_leaves))
        self.leaf_log_vars = nn.Parameter(torch.zeros(n_leaves) - 2.0)

    def forward(self, x):
        batch = x.size(0)
        leaf_probs = self._leaf_probs(x, batch)

        mean = (leaf_probs * self.leaf_means.unsqueeze(0)).sum(1)
        # mixture variance: E[X^2] - E[X]^2
        leaf_vars = self.leaf_log_vars.exp().unsqueeze(0)
        leaf_m2 = (self.leaf_means ** 2 + leaf_vars)
        var = (leaf_probs * leaf_m2).sum(1) - mean ** 2
        var = var.clamp(min=1e-6)
        return mean, var, leaf_probs

    def _leaf_probs(self, x, batch):
        node_p = {1: torch.ones(batch, device=x.device)}
        idx = 0
        for d in range(self.depth):
            for pos in range(2 ** d):
                node = 2 ** d + pos
                if node not in node_p:
                    continue
                logit = x @ self.node_weights[idx] + self.node_biases[idx]
                p_right = torch.sigmoid(logit)
                node_p[2 * node] = node_p[node] * (1.0 - p_right)
                node_p[2 * node + 1] = node_p[node] * p_right
                idx += 1
        leaf_start = 2 ** self.depth
        return torch.stack([node_p[leaf_start + i] for i in range(2 ** self.depth)], dim=1)


class TemporalNeuralTreeEnsemble(nn.Module):
    """
    GRU-encoded Neural Tree Ensemble with leaf-level uncertainty.

    Takes 30-cycle sequences as input (same as LSTM), routes through soft
    decision trees that output calibrated uncertainty bounds alongside
    gradient-based feature importance.
    """

    def __init__(self, input_dim: int = 17, hidden_dim: int = 64,
                 gru_layers: int = 2, n_trees: int = 5, depth: int = 5,
                 dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.GRU(
            input_dim, hidden_dim, gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.trees = nn.ModuleList([
            UncertainSoftDecisionTree(hidden_dim, depth) for _ in range(n_trees)
        ])
        self._y_min = 0.0
        self._y_max = 130.0

    def forward(self, x):
        out, _ = self.encoder(x)
        z = self.norm(out[:, -1, :])
        results = [tree(z) for tree in self.trees]
        means = torch.stack([r[0] for r in results], dim=1)
        vars_ = torch.stack([r[1] for r in results], dim=1)
        mean = means.mean(dim=1)
        # total uncertainty: mean of individual variances + variance of means
        aleatoric = vars_.mean(dim=1)
        epistemic = means.var(dim=1)
        total_var = aleatoric + epistemic
        return mean, total_var

    def predict_with_uncertainty(self, x):
        mean, var = self.forward(x)
        return mean, var.sqrt()

    def get_feature_importance(self, x: torch.Tensor) -> np.ndarray:
        """Input-gradient importance averaged over samples and timesteps."""
        self.eval()
        x = x.detach().clone().requires_grad_(True)
        mean, _ = self.forward(x)
        mean.sum().backward()
        imp = x.grad.abs().mean(dim=(0, 1)).detach().numpy()
        return imp / (imp.sum() + 1e-8)


def train_temporal_nt(model: TemporalNeuralTreeEnsemble,
                      X_train: np.ndarray, y_train: np.ndarray,
                      epochs: int = 200, lr: float = 5e-4,
                      batch_size: int = 512, sensor_dropout: float = 0.1,
                      verbose: bool = True) -> list:
    """
    Train with Gaussian NLL loss + sensor channel dropout augmentation.

    Sensor dropout zeros entire channels across all timesteps to simulate
    complete sensor failure — the practically important failure mode.
    """
    model._y_min = float(y_train.min())
    model._y_max = float(y_train.max())
    y_norm = (y_train - model._y_min) / (model._y_max - model._y_min + 1e-8)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_norm, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t),
        batch_size=batch_size, shuffle=True
    )

    history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            # Channel-level sensor dropout: zero entire sensor across all timesteps
            if sensor_dropout > 0:
                n_feat = xb.shape[2]
                mask = (torch.rand(xb.shape[0], 1, n_feat, device=xb.device)
                        > sensor_dropout).float()
                xb = xb * mask

            optimizer.zero_grad()
            mean, var = model(xb)
            # Gaussian NLL: 0.5 * [(y-mu)^2/sigma^2 + log(sigma^2)]
            loss = 0.5 * (((yb - mean) ** 2 / var) + var.log()).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / len(loader)
        history.append(avg)
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  TempNT Epoch {epoch+1:3d}/{epochs} | NLL: {avg:.4f}")

    return history


def predict_temporal_nt(model: TemporalNeuralTreeEnsemble,
                        X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mean_pred, std_pred) in original RUL scale."""
    model.eval()
    with torch.no_grad():
        mean, var = model(torch.tensor(X, dtype=torch.float32))
    scale = model._y_max - model._y_min
    mean_rul = mean.numpy() * scale + model._y_min
    std_rul = var.sqrt().numpy() * scale
    return mean_rul, std_rul
