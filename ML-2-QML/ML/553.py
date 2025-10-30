"""Enhanced classical regression model with feature scaling and deeper architecture.

- Uses a configurable multi‑layer perceptron with batch‑norm, dropout and optional
  feature expansion.
- Exposes a static metric evaluator for quick sanity checks.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic regression data inspired by a superposition of angles."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that returns a feature vector and a regression target."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel(nn.Module):
    """Deep MLP with configurable depth, dropout and feature expansion."""

    def __init__(
        self,
        num_features: int,
        hidden_units: list[int] | None = None,
        dropout: float = 0.1,
        expand_features: bool = False,
    ):
        super().__init__()
        if hidden_units is None:
            hidden_units = [64, 32, 16]

        layers = []
        current_in = num_features
        if expand_features:
            # simple polynomial expansion: add squared terms
            layers.append(nn.Linear(num_features, num_features * 2))
            layers.append(nn.ReLU())
            current_in = num_features * 2

        for hidden in hidden_units:
            layers.append(nn.Linear(current_in, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_in = hidden

        layers.append(nn.Linear(current_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)

    @staticmethod
    def evaluate(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
        """Return MSE and R² metrics for quick evaluation."""
        mse = torch.mean((y_true - y_pred) ** 2).item()
        ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        r2 = (1 - ss_res / ss_total).item() if ss_total!= 0 else 0.0
        return {"mse": mse, "r2": r2}


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
