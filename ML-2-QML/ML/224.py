"""Enhanced classical regression model with flexible architecture and data augmentation.

This module mirrors the original `QuantumRegression.py` but adds dropout,
batch normalization, and a residual connection to improve learning
performance on noisy data.  The dataset generation function remains
identical, providing reproducibility with the quantum counterpart.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data where the target is a smooth sinusoidal
    function of the sum of the input features.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch Dataset wrapping the synthetic regression data.
    """
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
    """
    Feed‑forward regression network with dropout, batchnorm and a residual skip.
    """
    def __init__(self, num_features: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Residual connection from input to output
        self.residual = nn.Linear(num_features, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.net(state_batch)
        res = self.residual(state_batch)
        return out + res


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
