"""
Explainable Neural Tree with MLP Embedding.

Architecture:
  Input → MLP Embedding → Soft Decision Tree Ensemble → RUL prediction

Key improvements:
- MLP prefix learns robust feature representations before tree routing
- Training-time sensor dropout teaches robustness to missing inputs
- Ensemble of trees for stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SoftDecisionTree(nn.Module):
    def __init__(self, input_dim: int, depth: int = 5):
        super().__init__()
        self.depth = depth
        n_internal = 2 ** depth - 1
        n_leaves = 2 ** depth

        self.node_weights = nn.Parameter(torch.randn(n_internal, input_dim) * 0.01)
        self.node_biases = nn.Parameter(torch.zeros(n_internal))
        self.leaf_values = nn.Parameter(torch.linspace(0, 1, n_leaves))

    def forward(self, x):
        batch = x.size(0)
        leaf_probs = self._leaf_probs(x, batch)
        pred = (leaf_probs * self.leaf_values.unsqueeze(0)).sum(dim=1)
        return pred, leaf_probs

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


class NeuralTreeEnsemble(nn.Module):
    """
    Neural Tree Ensemble with MLP embedding.

    The MLP learns compact feature representations; each Soft Decision Tree
    uses these representations for interpretable, calibrated RUL routing.
    Feature importance is derived from input gradients through the full model.
    """

    def __init__(self, input_dim: int, n_trees: int = 5, depth: int = 5,
                 hidden_dim: int = 32, dropout: float = 0.15):
        super().__init__()

        # MLP embedding: raw sensors → learned representation
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.trees = nn.ModuleList([
            SoftDecisionTree(hidden_dim, depth) for _ in range(n_trees)
        ])
        self.dropout = nn.Dropout(dropout)
        self._y_min = 0.0
        self._y_max = 130.0

    def forward(self, x):
        z = self.embedding(x)
        preds = [tree(z)[0] for tree in self.trees]
        return torch.stack(preds, dim=1).mean(dim=1)

    def get_feature_importance(self, x: torch.Tensor) -> np.ndarray:
        """Input-gradient feature importance (averaged over samples)."""
        self.eval()
        x = x.detach().clone().requires_grad_(True)
        pred = self.forward(x)
        pred.sum().backward()
        imp = x.grad.abs().mean(dim=0).detach().numpy()
        return imp / (imp.sum() + 1e-8)


def train_neural_tree(model: NeuralTreeEnsemble, X_train: np.ndarray,
                      y_train: np.ndarray, epochs: int = 200, lr: float = 5e-4,
                      batch_size: int = 512, sensor_dropout: float = 0.1,
                      verbose: bool = True) -> list:
    """
    Train with:
    - y normalization for stable gradients
    - Sensor dropout augmentation for missing-sensor robustness
    - Cosine annealing LR schedule
    """
    model._y_min = float(y_train.min())
    model._y_max = float(y_train.max())
    y_norm = (y_train - model._y_min) / (model._y_max - model._y_min + 1e-8)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_norm, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    n_features = X_train.shape[1]
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            # Sensor dropout augmentation — randomly zero sensors during training
            if sensor_dropout > 0:
                mask = (torch.rand(xb.shape, device=xb.device) > sensor_dropout).float()
                xb = xb * mask

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
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg:.6f}")

    return history


def predict_neural_tree(model: NeuralTreeEnsemble, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return pred * (model._y_max - model._y_min) + model._y_min
