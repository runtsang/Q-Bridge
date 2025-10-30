"""Hybrid classical regression model combining feature extraction and a sampler network."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample data from a simple sinusoidal superposition:
    y = sin(θ) + 0.1 * cos(2θ), where θ is the sum of feature values.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset yielding (features, target) pairs for regression."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class SamplerQNN(nn.Module):
    """
    Simple classical sampler network that maps a 2‑dimensional vector
    to a probability distribution over two outcomes.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

class HybridRegression(nn.Module):
    """
    Classical hybrid regression model:
    * Feature extractor (MLP) transforms raw inputs.
    * Classical sampler network produces a probability distribution.
    * Linear head maps probabilities to a scalar output.
    """
    def __init__(self, num_features: int, hidden_dim: int = 32, sampler_dim: int = 2):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.sampler = SamplerQNN()
        self.head = nn.Linear(sampler_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature_extractor(x)
        probs = self.sampler(h)
        return self.head(probs).squeeze(-1)

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data", "SamplerQNN"]
