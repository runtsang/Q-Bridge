"""Hybrid regression model combining classical encoding with a sampler-based augmentation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data using a superposition-inspired function."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

def SamplerQNN():
    """Classical sampler network providing a probability distribution over two outputs."""
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return F.softmax(self.net(inputs), dim=-1)

    return SamplerModule()

class HybridRegressionModel(nn.Module):
    """Classical regression model that encodes inputs, augments with a sampler, and predicts a scalar."""
    def __init__(self, num_features: int):
        super().__init__()
        # Encoder reduces dimensionality to 2 for sampler compatibility
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        self.sampler = SamplerQNN()
        self.head = nn.Linear(2, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(state_batch)
        sampled = self.sampler(encoded)
        return self.head(sampled).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data", "SamplerQNN"]
