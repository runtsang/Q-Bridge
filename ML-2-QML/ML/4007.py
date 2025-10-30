"""Hybrid classical regression model that combines an MLP with a random linear layer and batch normalisation.

The model inherits the superposition data generation from the original regression seed,
while adding a classical random linear layer and batch‑normalisation inspired by
Quantum‑NAT to improve feature diversity and stability.

The implementation is fully compatible with PyTorch and can be used as a drop‑in
replacement for the original ``QModel`` in the test harness.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data with a superposition‑like structure.

    Parameters
    ----------
    num_features : int
        Dimensionality of each sample.
    samples : int
        Number of samples to generate.

    Returns
    -------
    x : np.ndarray
        Feature matrix of shape (samples, num_features).
    y : np.ndarray
        Regression targets of shape (samples,).
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset wrapper that returns features and targets in a dictionary
    compatible with the training loop.
    """
    def __init__(self, samples: int, num_features: int, reshape_to_2d: bool = False):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        if reshape_to_2d:
            # Reshape to a square image for compatibility with CNN encoders
            side = int(np.sqrt(num_features))
            self.features = self.features.reshape(-1, 1, side, side)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridRegressionModel(nn.Module):
    """
    Classical regression model that mimics the quantum variant but
    uses an MLP with a random linear layer and batch normalisation.
    """
    def __init__(self, num_features: int, hidden_dims: tuple[int,...] = (32, 16)):
        super().__init__()
        layers = []
        in_dim = num_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.feature_extractor = nn.Sequential(*layers)

        # Non‑trainable random linear layer to inject noise
        self.random_linear = nn.Linear(in_dim, in_dim, bias=False)
        self.random_linear.weight.requires_grad = False
        nn.init.orthogonal_(self.random_linear.weight)

        self.norm = nn.BatchNorm1d(in_dim)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(state_batch)
        x = self.random_linear(x)
        x = self.norm(x)
        return self.head(x).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
