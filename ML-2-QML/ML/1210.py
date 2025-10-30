"""Classical regression module with residual dense blocks and noise schedule."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import OrderedDict

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for regression, mirroring the seed but with an added noise schedule."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    # Add a sinusoidal noise term that decays with sample index
    noise = 0.1 * np.cos(2 * angles) * np.exp(-0.01 * np.arange(samples))
    y = np.sin(angles) + 0.1 * np.cos(2 * angles) + noise
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that yields a dictionary with features and target."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ResidualBlock(nn.Module):
    """A small residual dense block that can be stacked."""
    def __init__(self, dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(dim, hidden)),
                    ("act1", nn.ReLU()),
                    ("fc2", nn.Linear(hidden, dim)),
                    ("act2", nn.ReLU()),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class QuantumRegression(nn.Module):
    """Classical regression network with residual stack."""
    def __init__(self, num_features: int, residual_depth: int = 2):
        super().__init__()
        self.input_dim = num_features
        self.base = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.residuals = nn.ModuleList([ResidualBlock(16) for _ in range(residual_depth)])
        self.head = nn.Linear(16, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        x = self.base(state_batch)
        for res in self.residuals:
            x = res(x)
        return self.head(x).squeeze(-1)

__all__ = ["QuantumRegression", "RegressionDataset", "generate_superposition_data"]
