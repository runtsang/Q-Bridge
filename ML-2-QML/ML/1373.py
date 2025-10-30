"""Classical regression model using a deep MLP.

This module mirrors the seed but expands the architecture with a
feature‑extractor and a more expressive head.  The dataset and data
generation logic are unchanged; the model can be trained with any
standard PyTorch training loop.

The class name is ``QuantumRegression__gen332`` so that it can be
imported from both the classical and quantum modules.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data with a superposition‑like target."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset returning raw features and target values."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumRegression__gen332(nn.Module):
    """Deep MLP for regression on the synthetic dataset."""
    def __init__(self, num_features: int, hidden_dims: tuple[int,...] = (64, 32, 16)):
        super().__init__()
        layers = []
        in_dim = num_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)

__all__ = ["QuantumRegression__gen332", "RegressionDataset", "generate_superposition_data"]
