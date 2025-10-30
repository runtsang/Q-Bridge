"""Enhanced classical regression model with feature scaling, dropout, and batch normalization."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data based on a superposition-inspired mapping."""
    # Uniformly sample features in [-1, 1]
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    # Construct labels with a non-linear function plus periodic modulation
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles) + 0.05 * np.random.randn(samples)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """PyTorch Dataset that normalises features to zero mean and unit variance."""

    def __init__(self, samples: int, num_features: int):
        raw_features, self.labels = generate_superposition_data(num_features, samples)
        # Feature scaling
        self.mean = raw_features.mean(axis=0, keepdims=True)
        self.std = raw_features.std(axis=0, keepdims=True) + 1e-8
        self.features = (raw_features - self.mean) / self.std

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class RegressionModel(nn.Module):
    """A deeper MLP with dropout and batchnorm for regression."""

    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)


__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data"]
