"""Classical regression dataset and hybrid residual network."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data where the target is a higher‑order function
    of the feature sum, providing a richer regression signal than the
    original sinusoidal toy function.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    sums = x.sum(axis=1)
    y = np.sin(sums) + 0.5 * sums ** 2 + 0.05 * np.random.randn(samples)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset returning feature vectors and scalar targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class SharedRegressionModel(nn.Module):
    """Residual neural network with dropout for regression."""
    def __init__(self, num_features: int, hidden_sizes: list[int] | None = None, dropout: float = 0.1):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 32, 16]
        layers = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch).squeeze(-1)


__all__ = ["SharedRegressionModel", "RegressionDataset", "generate_superposition_data"]
