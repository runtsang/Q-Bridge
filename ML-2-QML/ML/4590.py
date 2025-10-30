"""Hybrid classical regression model with optional RBF kernel feature extraction.

This module defines `HybridRegressionModel`, a lightweight feed‑forward
network that optionally applies an RBF kernel to the input features.
The dataset utilities are identical to those in the original
`QuantumRegression.py` but return real‑valued feature vectors.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for regression with sinusoidal pattern."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that returns a dictionary of states and targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridRegressionModel(nn.Module):
    """Hybrid classical regression model with optional RBF kernel feature map."""
    def __init__(self, num_features: int, use_rbf: bool = False, gamma: float = 1.0):
        super().__init__()
        self.use_rbf = use_rbf
        self.gamma = gamma
        self.num_features = num_features
        input_dim = 1 if use_rbf else num_features
        self.head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        if self.use_rbf:
            # Compute RBF for each sample independently against a random prototype
            proto = torch.randn(self.num_features, device=state_batch.device)
            dists = torch.cdist(state_batch, proto.unsqueeze(0)) ** 2
            features = torch.exp(-self.gamma * dists)
        else:
            features = state_batch
        return self.head(features).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
