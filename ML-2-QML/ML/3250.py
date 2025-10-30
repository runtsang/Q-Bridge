"""Hybrid sampler and regression model implemented purely in PyTorch.

The module reproduces the SamplerQNN architecture and augments it with a
regression head.  It also includes a lightweight dataset generator
mirroring the quantum regression example for consistency.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for regression.

    The data mimics the superposition function used in the quantum
    regression example:  y = sin(θ) + 0.1*cos(2θ).
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

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

class HybridSamplerRegressor(nn.Module):
    """Classical sampler network followed by a regression head."""
    def __init__(self, latent_dim: int = 2, num_features: int = 2):
        super().__init__()
        # Sampler: maps latent vector to a probability distribution over features
        self.sampler = nn.Sequential(
            nn.Linear(latent_dim, 4),
            nn.Tanh(),
            nn.Linear(4, num_features),
        )
        # Regression head: mirrors the architecture from the quantum example
        self.regressor = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        probs = F.softmax(self.sampler(z), dim=-1)
        return self.regressor(probs).squeeze(-1)

__all__ = ["HybridSamplerRegressor", "RegressionDataset", "generate_superposition_data"]
