"""ML module for hybrid quantum regression.

Provides data generation, a classical regression head and utilities that mirror the
original QuantumRegression.py while extending the feature set to support a quantum encoder
and a quanvolution filter."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic dataset where each sample is a linear combination of
    features mapped through a simple trigonometric function.  This mimics the
    quantum superposition labels in the original seed but keeps the data purely
    classical for the ML branch.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that returns raw features and the synthetic target."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ClassicalRegressor(nn.Module):
    """Simple MLP that mirrors the original QModel but can be extended."""
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)

__all__ = ["generate_superposition_data", "RegressionDataset", "ClassicalRegressor"]
