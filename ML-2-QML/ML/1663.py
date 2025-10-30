"""Enhanced classical regression model with residual blocks and dropout."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a toy dataset that mimics a superposition‑like signal."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset yielding feature vectors and regression targets."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class ResidualBlock(nn.Module):
    """A simple residual block with batch‑norm and dropout."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(0.1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)
        return x + out  # residual connection


class SharedClassName(nn.Module):
    """Extended regression network with residual layers."""

    def __init__(self, num_features: int, hidden_dim: int = 64, depth: int = 3):
        super().__init__()
        layers = [
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        ]
        for _ in range(depth):
            layers.append(ResidualBlock(hidden_dim))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)


__all__ = ["SharedClassName", "RegressionDataset", "generate_superposition_data"]
