"""Hybrid classical regression model.

Provides a dataset generator, a Dataset wrapper and a residual feed‑forward
network that can be trained on the superposition data.  The model is
designed to have the same public interface as the quantum counterpart
(`HybridRegressionModel`) so that experiments can be run in parallel
without changing the calling code.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data.

    Parameters
    ----------
    num_features : int
        Number of input features.
    samples : int
        Number of samples to generate.

    Returns
    -------
    x : np.ndarray
        Feature matrix of shape (samples, num_features).
    y : np.ndarray
        Target vector of shape (samples,).
    """
    rng = np.random.default_rng()
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset wrapping the synthetic superposition data."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class HybridRegressionModel(nn.Module):
    """
    Residual feed‑forward network for regression.

    The architecture mirrors the quantum head: a linear layer that maps
    the input dimension to an intermediate size, followed by a
    residual block and a final linear regression head.
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
        )
        self.residual = nn.Linear(num_features, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.net(states)
        res = self.residual(states)
        x = x + res
        return self.head(x).squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
