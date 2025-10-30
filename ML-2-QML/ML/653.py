"""Enhanced classical regression model with residual blocks and dropout."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int, noise_std: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset inspired by quantum superposition.
    The features are uniformly sampled in [-1, 1] and the target is a noisy
    combination of sin and cos of the feature sum.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    y += np.random.normal(scale=noise_std, size=y.shape).astype(np.float32)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset wrapping the synthetic superposition data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class ResidualBlock(nn.Module):
    """Simple residual block with linear, batch norm and ReLU."""
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.fc(x)))


class QModel(nn.Module):
    """Classical regression model with residual connections and dropout."""
    def __init__(self, num_features: int):
        super().__init__()
        self.input_layer = nn.Linear(num_features, 64)
        self.blocks = nn.ModuleList([ResidualBlock(64) for _ in range(3)])
        self.dropout = nn.Dropout(p=0.2)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.input_layer(state_batch)
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual
        x = self.dropout(x)
        return self.output_layer(x).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
