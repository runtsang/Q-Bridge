"""Enhanced classical regression model with dropout and batch‑norm."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int, noise: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data based on a superposition target function."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise > 0.0:
        y += noise * np.random.randn(samples).astype(np.float32)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset yielding feature tensors and regression targets."""

    def __init__(self, samples: int, num_features: int, noise: float = 0.0):
        self.features, self.labels = generate_superposition_data(num_features, samples, noise)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel(nn.Module):
    """Feed‑forward network with batch‑norm and dropout for robust regression."""

    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
