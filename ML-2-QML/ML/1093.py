"""Hybrid classical regression model with residual learning and feature‑wise scaling."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for regression.

    Each sample is a vector of `num_features` random angles in [0, 2π). The target is a
    nonlinear function of the sum of angles: sin(2 * sum) * cos(sum).
    """
    rng = np.random.default_rng()
    angles = rng.uniform(0, 2 * np.pi, size=(samples, num_features))
    features = angles
    sum_angles = angles.sum(axis=1)
    labels = np.sin(2 * sum_angles) * np.cos(sum_angles)
    return features.astype(np.float32), labels.astype(np.float32)


class RegressionDataset(Dataset):
    """
    PyTorch dataset that yields a feature vector and a target scalar.
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


class ResidualBlock(nn.Module):
    """
    A simple residual block that preserves the input dimension.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out + residual


class QuantumRegression__gen019(nn.Module):
    """
    Residual‑dense regression network that can be trained jointly with a quantum module.
    """
    def __init__(self, num_features: int, hidden_dim: int = 64, num_blocks: int = 3):
        super().__init__()
        self.initial = nn.Linear(num_features, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.final = nn.Linear(hidden_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        x = self.initial(state_batch)
        for block in self.blocks:
            x = block(x)
        return self.final(x).squeeze(-1)


__all__ = ["QuantumRegression__gen019", "RegressionDataset", "generate_superposition_data"]
