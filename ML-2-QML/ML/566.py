"""Enhanced classical regression model with residual connections and feature importance."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic regression dataset mimicking a quantum superposition."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    # Add a small random noise to make the task non‑trivial
    y += 0.05 * np.random.randn(samples).astype(np.float32)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that returns floating point states and target values."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel(nn.Module):
    """Residual neural network with dropout for regression."""
    def __init__(self, num_features: int, hidden_dims: list[int] | None = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]
        layers = []
        in_dim = num_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(h, in_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        self.res_blocks = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = state_batch.to(torch.float32)
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual
        return self.out(x).squeeze(-1)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(X)

    def evaluate(self, X: torch.Tensor, y_true: torch.Tensor) -> dict[str, float]:
        preds = self.predict(X)
        mse = mean_squared_error(y_true.cpu().numpy(), preds.cpu().numpy())
        mae = mean_absolute_error(y_true.cpu().numpy(), preds.cpu().numpy())
        return {"mse": mse, "mae": mae}

    def feature_importance(self, X: torch.Tensor, y_true: torch.Tensor, n_repeats: int = 5) -> np.ndarray:
        """Permutation importance on the trained model."""
        baseline = self.evaluate(X, y_true)["mse"]
        importances = []
        X_np = X.detach().cpu().numpy()
        for i in range(X.shape[1]):
            imp = 0.0
            for _ in range(n_repeats):
                X_perm = X_np.copy()
                np.random.shuffle(X_perm[:, i])
                imp += self.evaluate(torch.tensor(X_perm, device=X.device), y_true)["mse"]
            imp /= n_repeats
            importances.append(imp - baseline)
        return np.array(importances)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
