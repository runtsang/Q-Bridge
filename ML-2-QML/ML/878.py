"""Enhanced classical regression model with residual blocks and dropout.

The module mirrors the structure of the original quantum implementation
while adding depth and regularisation to improve generalisation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data based on a superposition‑inspired function.
    The input is a vector of angles; the target is
    ``sin(2*theta) * cos(phi)`` where ``theta`` and ``phi`` are the first two
    components of the input vector.
    """
    x = np.random.uniform(0, 2 * np.pi, size=(samples, num_features)).astype(np.float32)
    thetas = x[:, 0]
    phis = x[:, 1]
    y = np.sin(2 * thetas) * np.cos(phis)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset yielding angle vectors and regression targets."""

    def __init__(self, samples: int, num_features: int = 2):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class ResidualBlock(nn.Module):
    """Simple residual block with linear layer, batch‑norm and ReLU."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.linear(x) + x)


class QuantumRegressionModel(nn.Module):
    """
    Classical regression network that emulates the depth of the quantum
    architecture: two residual blocks followed by a linear head.
    """

    def __init__(self, num_features: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(num_features),
            ResidualBlock(num_features),
            nn.Dropout(p=0.2),
            nn.Linear(num_features, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch).squeeze(-1)


__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data"]
