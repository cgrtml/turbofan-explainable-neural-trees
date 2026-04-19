"""
Soft Neural Decision Tree for Regression
Interpretable tree structure with learned soft routing functions.
Based on: Frosst & Hinton (2017) "Distilling a Neural Network Into a Soft Decision Tree"
Adapted for regression (RUL prediction).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SoftDecisionTree(nn.Module):
    """
    Soft Neural Decision Tree for regression.

    Each internal node learns a soft binary routing using a linear layer + sigmoid.
    Leaf nodes hold learned output values (RUL predictions).
    Path probabilities are used for explainability.
    """

    def __init__(self, input_dim: int, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.depth = depth
        self.input_dim = input_dim

        n_internal = 2 ** depth - 1
        n_leaves = 2 ** depth

        # Each internal node: linear layer mapping input -> routing probability
        self.node_weights = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(n_internal)
        ])
        self.node_batchnorms = nn.ModuleList([
            nn.BatchNorm1d(input_dim) for _ in range(n_internal)
        ])

        # Leaf values (learnable)
        self.leaf_values = nn.Parameter(torch.zeros(n_leaves))

        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.node_weights:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.leaf_values, 0, 1)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        x = self.dropout(x)

        # Path probabilities for all leaves — shape: (batch, n_leaves)
        leaf_probs = self._compute_leaf_probs(x, batch_size)

        # Weighted sum of leaf values
        pred = (leaf_probs * self.leaf_values.unsqueeze(0)).sum(dim=1)
        return pred, leaf_probs

    def _compute_leaf_probs(self, x: torch.Tensor, batch_size: int):
        n_leaves = 2 ** self.depth
        # Start with probability 1 for root
        node_probs = [None] * (2 ** (self.depth + 1))
        node_probs[1] = torch.ones(batch_size, device=x.device)

        node_idx = 0
        for d in range(self.depth):
            for pos in range(2 ** d):
                current = 2 ** d + pos  # 1-indexed node
                if node_probs[current] is None:
                    continue

                # Routing probability (go right)
                bn_x = self.node_batchnorms[node_idx](x) if batch_size > 1 else x
                p_right = torch.sigmoid(self.node_weights[node_idx](bn_x).squeeze(1))
                p_left = 1.0 - p_right

                left_child = 2 * current
                right_child = 2 * current + 1

                node_probs[left_child] = node_probs[current] * p_left
                node_probs[right_child] = node_probs[current] * p_right

                node_idx += 1

        # Collect leaf probabilities
        leaf_start = 2 ** self.depth
        leaf_probs = torch.stack(
            [node_probs[leaf_start + i] for i in range(n_leaves)], dim=1
        )
        return leaf_probs

    def get_feature_importance(self, x: torch.Tensor) -> np.ndarray:
        """Compute input gradient-based feature importance."""
        self.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        pred, _ = self.forward(x_tensor)
        pred.sum().backward()
        importance = x_tensor.grad.abs().mean(dim=0).detach().numpy()
        return importance / (importance.sum() + 1e-8)


class NeuralTreeEnsemble(nn.Module):
    """Ensemble of Soft Decision Trees for improved robustness."""

    def __init__(self, input_dim: int, n_trees: int = 3, depth: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.trees = nn.ModuleList([
            SoftDecisionTree(input_dim, depth, dropout) for _ in range(n_trees)
        ])

    def forward(self, x: torch.Tensor):
        preds = []
        for tree in self.trees:
            pred, _ = tree(x)
            preds.append(pred)
        ensemble_pred = torch.stack(preds, dim=1).mean(dim=1)
        return ensemble_pred

    def get_feature_importance(self, x: torch.Tensor) -> np.ndarray:
        importances = []
        for tree in self.trees:
            imp = tree.get_feature_importance(x)
            importances.append(imp)
        return np.mean(importances, axis=0)


def train_neural_tree(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                      epochs: int = 100, lr: float = 1e-3, batch_size: int = 256,
                      verbose: bool = True):
    """Train Neural Tree with MSE loss + L2 regularization."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            if isinstance(model, NeuralTreeEnsemble):
                pred = model(xb)
            else:
                pred, _ = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        history.append(avg_loss)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f}")

    return history


def predict_neural_tree(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Run inference on Neural Tree model."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        if isinstance(model, NeuralTreeEnsemble):
            pred = model(X_t)
        else:
            pred, _ = model(X_t)
    return pred.numpy()
