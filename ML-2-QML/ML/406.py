"""Classical regression model with dropout and a deep MLP."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data that mimics a quantum superposition encoding.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    # Add small Gaussian noise for realism.
    y += 0.05 * np.random.randn(samples)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset yielding feature vectors and targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class QuantumRegressionModel(nn.Module):
    """
    Classical neural network with dropout and a multi‑layer perceptron.
    """
    def __init__(self, num_features: int, hidden_sizes: list[int] | None = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 32, 16]
        layers = []
        in_features = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_features = h
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data"]
