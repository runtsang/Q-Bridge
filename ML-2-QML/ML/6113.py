"""Enhanced classical regression model with residual blocks and dropout."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int, noise: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data based on a superposition-like pattern.
    The target is a non‑linear combination of the input angles with added Gaussian noise.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles) + noise * np.random.randn(samples)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch dataset that normalises inputs and caches tensors for efficient loading.
    """
    def __init__(self, samples: int, num_features: int, noise: float = 0.05):
        self.features, self.labels = generate_superposition_data(num_features, samples, noise)
        # Standardise features
        self.mean = self.features.mean(axis=0)
        self.std = self.features.std(axis=0) + 1e-6
        self.features = (self.features - self.mean) / self.std

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class ResidualBlock(nn.Module):
    """
    Simple residual block with batch normalisation and dropout.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.linear(x)
        out = self.bn(out)
        out = torch.relu(out)
        out = self.dropout(out)
        return torch.relu(out + residual)


class QModel(nn.Module):
    """
    Deep feed‑forward network with residual connections for regression.
    """
    def __init__(self, num_features: int, hidden_dim: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(num_features, hidden_dim), nn.ReLU()]
        for _ in range(depth):
            layers.append(ResidualBlock(hidden_dim))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
