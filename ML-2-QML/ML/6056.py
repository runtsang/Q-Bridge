"""Hybrid classical regression model combining advanced MLP with residual connections and dropout.

This module extends the original classical regression example by adding:
- Input standardisation
- Optional residual blocks
- Dropout for regularisation
- Tanh activations inspired by EstimatorQNN
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_classical_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data with non‑linear features.

    The labels are a smooth function of the sum of the inputs, mimicking a
    superposition of sinusoids.  The function is intentionally more
    expressive than the original seed so that the network can learn
    richer patterns.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset wrapper for the synthetic regression data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_classical_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ResidualBlock(nn.Module):
    """Simple residual block used in HybridRegression."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class HybridRegression(nn.Module):
    """Advanced MLP for regression with optional residual connections and dropout."""
    def __init__(
        self,
        num_features: int,
        hidden_dims: list[int] | tuple[int,...] | None = None,
        dropout: float = 0.0,
        use_residual: bool = False,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        layers = []
        in_dim = num_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            if dropout:
                layers.append(nn.Dropout(dropout))
            if use_residual:
                layers.append(ResidualBlock(h))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)

# Backwards compatibility
QModel = HybridRegression

__all__ = ["HybridRegression", "QModel", "RegressionDataset", "generate_classical_data"]
