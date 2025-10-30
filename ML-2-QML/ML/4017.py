"""Enhanced classical estimator that mirrors the quantum example but with a deeper network and residual connections."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset yielding feature vectors and target values derived from superposition states."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class EstimatorQNNGen338(nn.Module):
    """
    A classical feed‑forward regressor with residual connections and dropout.
    Designed to be a drop‑in replacement for the original EstimatorQNN.
    """
    def __init__(self, num_features: int = 2, hidden_dim: int = 64) -> None:
        super().__init__()
        # Input layer
        self.in_layer = nn.Linear(num_features, hidden_dim)
        # Hidden layers with batch norm, ReLU, dropout
        self.hidden = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # Output layer
        self.out_layer = nn.Linear(hidden_dim, 1)
        # Residual mapping to match input dimension
        self.residual = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_layer(x)
        h = self.hidden(h)
        out = self.out_layer(h)
        # Residual addition
        return out + self.residual(x).squeeze(-1)

__all__ = ["EstimatorQNNGen338", "RegressionDataset", "generate_superposition_data"]
