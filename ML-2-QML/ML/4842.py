"""Hybrid regression model – classical implementation.

The module reproduces the synthetic regression dataset from the original
seed while adding a CNN encoder that prepares features for a linear
regression head.  The architecture mirrors the Quantum‑NAT encoder
(2 convolutional layers + pooling) and the Quantum‑Regression network,
but stays fully classical (PyTorch only).

The public API matches the original ``QuantumRegression.py`` so that
existing scripts can import this file without modification.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data.

    The data emulate a quantum superposition of angles:  
    ``y = sin(θ) + 0.1*cos(2θ)`` where ``θ`` is the sum of input
    features.  This matches the original seed but is refactored into
    a reusable function.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset wrapping the synthetic regression data."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {"states": torch.tensor(self.features[index], dtype=torch.float32),
                "target": torch.tensor(self.labels[index], dtype=torch.float32)}


class HybridRegressionModel(nn.Module):
    """Classical CNN‑based regression head.

    The encoder follows the same pattern as the Quantum‑NAT
    convolutional block (two conv layers + pooling) and outputs a
    64‑dimensional feature vector.  A final linear layer maps to a
    scalar target.  Batch‑normalisation is applied to the output for
    numerical stability, mirroring the quantum variant.
    """

    def __init__(self, num_features: int = 4):
        super().__init__()
        # Encoder – lightweight 2‑layer CNN
        self.encoder = nn.Sequential(
            nn.Conv1d(num_features, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.flatten = nn.Flatten()
        # Regression head
        self.head = nn.Sequential(
            nn.Linear(32 * (num_features // 4), 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.BatchNorm1d(1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Input shape: (batch, features)
        x = state_batch.unsqueeze(1)  # (batch, 1, features)
        features = self.encoder(x)
        flat = self.flatten(features)
        out = self.head(flat).squeeze(-1)
        return out


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
