"""Classical regression model with residual connections and data generation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_regression_data(num_features: int, samples: int, noise_std: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data with a sinusoidal target."""
    X = np.random.uniform(-np.pi, np.pi, size=(samples, num_features)).astype(np.float32)
    angles = X.sum(axis=1)
    y = np.sin(angles) + noise_std * np.random.randn(samples)
    return X, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset returning features and target."""
    def __init__(self, samples: int, num_features: int, noise_std: float = 0.05):
        self.X, self.y = generate_regression_data(num_features, samples, noise_std)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.X)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

class ResidualBlock(nn.Module):
    """Simple residual block with batch norm and dropout."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))

class RegressionModel(nn.Module):
    """Deep residual network for regression."""
    def __init__(self, num_features: int, hidden_dim: int = 64, num_residual: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = [nn.Linear(num_features, hidden_dim), nn.ReLU()]
        for _ in range(num_residual):
            layers.append(ResidualBlock(hidden_dim, dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

__all__ = ["RegressionModel", "RegressionDataset", "generate_regression_data"]
