"""Enhanced classical regression model with regularization and optional noise.

Provides a flexible dataset generator that supports Gaussian noise, and a neural
network with batch normalization and dropout for robust learning.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(
    num_features: int, samples: int, noise_level: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data with optional Gaussian noise."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise_level > 0.0:
        y += np.random.normal(scale=noise_level, size=y.shape)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Torch Dataset wrapping the synthetic data."""

    def __init__(
        self,
        samples: int,
        num_features: int,
        noise_level: float = 0.0,
    ):
        self.features, self.labels = generate_superposition_data(
            num_features, samples, noise_level
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel(nn.Module):
    """Feed‑forward neural network with BatchNorm and Dropout."""

    def __init__(self, num_features: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
