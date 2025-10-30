"""Extended classical regression model with a learnable encoder and deeper MLP."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(
    num_features: int, samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate features that mimic the quantum superposition angles."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = 0.5 * np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset yielding raw feature vectors and target values."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumRegression(nn.Module):
    """Classical regression model with a learnable encoder and a multi‑layer perceptron."""

    def __init__(self, num_features: int):
        super().__init__()
        self.encoder = nn.Linear(num_features, 64)
        self.mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        x = self.encoder(state_batch)
        return self.mlp(x).squeeze(-1)


__all__ = ["QuantumRegression", "RegressionDataset", "generate_superposition_data"]
