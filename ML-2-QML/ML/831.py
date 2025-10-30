"""Enhanced classical regression model with residual connections and data augmentation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data that mimics the superposition‑based labels with data augmentation.

    A random orthogonal rotation is applied to the feature vectors to simulate
    small perturbations in the angles, improving generalisation.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    # Random orthogonal rotation
    rot = np.linalg.qr(np.random.randn(num_features, num_features))[0]
    x_aug = x @ rot
    angles = x_aug.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x_aug, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that yields batches of states and their target values."""

    def __init__(self, samples: int, num_features: int):
        super().__init__()
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ResidualMLP(nn.Module):
    """Three‑layer MLP with a residual skip‑connection from input to output."""

    def __init__(self, num_features: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.skip = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)

class QuantumRegressionEnhanced(nn.Module):
    """Hybrid classical regression model that can be trained with a standard PyTorch loop."""

    def __init__(self, num_features: int, hidden_dim: int = 32):
        super().__init__()
        self.model = ResidualMLP(num_features, hidden_dim)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)

__all__ = ["QuantumRegressionEnhanced", "RegressionDataset", "generate_superposition_data"]
