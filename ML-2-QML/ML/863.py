"""Hybrid classical regression model with residual connections and dropout."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data mimicking a quantum superposition.

    The returned features are real numbers in [-1, 1].  The target is a
    non‑linear function of the sum of the features, adding a small
    cosine perturbation to mimic interference.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
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


class QuantumRegressionModel(nn.Module):
    """A classical MLP with a residual path and dropout."""

    def __init__(self, num_features: int, hidden_sizes: tuple[int,...] = (64, 32), dropout: float = 0.1):
        super().__init__()
        layers = []
        input_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h
        self.main = nn.Sequential(*layers)
        self.residual = nn.Linear(num_features, input_dim)
        self.output = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_out = self.main(x)
        res_out = self.residual(x)
        out = main_out + res_out
        return self.output(out).squeeze(-1)


__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data"]
