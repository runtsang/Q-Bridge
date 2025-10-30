"""Classical regression model with residual connections and dropout.

This module extends the original simple feed‑forward architecture
by adding batch‑normalisation, a residual skip connection, and
dropout to improve generalisation on noisy superposition data.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a noisy sinusoidal regression target.

    The data mirrors the quantum seed but includes Gaussian noise
    to simulate measurement uncertainty.  The function is kept
    deterministic via a fixed random seed for reproducibility.
    """
    rng = np.random.default_rng(seed=42)
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    noise = rng.normal(0.0, 0.05, size=y.shape).astype(np.float32)
    return x, (y + noise).astype(np.float32)


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


class QModel(nn.Module):
    """Deep residual regression network with dropout."""

    def __init__(self, num_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        # Residual connection from input to the first hidden layer
        self.residual = nn.Linear(num_features, 64, bias=False)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.network(state_batch)
        res = self.residual(state_batch)
        # Combine residual and main path
        out += res
        return out.squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
