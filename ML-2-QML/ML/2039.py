"""Enhanced classical regression model with optional regularisation and noise-aware data generation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data"]

def generate_superposition_data(num_features: int, samples: int, noise_std: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data mimicking a superposition‑like feature pattern.
    Adds Gaussian noise to the target if noise_std > 0."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    if noise_std > 0.0:
        y += np.random.normal(0.0, noise_std, size=y.shape)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset wrapper for regression data."""
    def __init__(self, samples: int, num_features: int, noise_std: float = 0.0):
        self.features, self.labels = generate_superposition_data(num_features, samples, noise_std)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict:
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumRegressionModel(nn.Module):
    """Classical neural network for regression with optional dropout and weight decay."""
    def __init__(self, num_features: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch).squeeze(-1)
