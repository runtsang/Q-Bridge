"""Hybrid regression model with optional Fourier feature augmentation.

The original seed defined a purely classical MLP.  This extension keeps the same
``generate_superposition_data`` routine and ``RegressionDataset`` but adds a new
``QuantumRegression__gen399`` that can be instantiated with or without a
quantum‑like feature extractor.  The quantum part is a lightweight variational
circuit that produces a feature vector of the same dimensionality as the
classical embedding.  The two feature streams are concatenated before the final
linear layer, giving a richer representation for the regression target.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Data generation – unchanged from the seed
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data that can be used in both quantum and classical
    contexts.  The function returns two NumPy arrays: ``x`` (the feature
    vectors) and ``y`` (the target values).  The targets are a noisy
    sinusoidal function of the sum of the input features.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

# --------------------------------------------------------------------------- #
# Dataset wrapper
# --------------------------------------------------------------------------- #
class RegressionDataset(Dataset):
    """Dataset that returns states and targets as tensors."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Feature extractor – optional quantum‑like augmentation
# --------------------------------------------------------------------------- #
class RandomFourierFeatures(nn.Module):
    """Random Fourier feature mapping used to mimic a quantum feature map."""
    def __init__(self, in_dim: int, out_dim: int = 64, sigma: float = 1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_dim, out_dim) * sigma, requires_grad=False)
        self.b = nn.Parameter(torch.randn(out_dim) * 2 * np.pi, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, in_dim)
        projection = x @ self.W + self.b
        return torch.cos(projection)  # shape: (batch, out_dim)

# --------------------------------------------------------------------------- #
# Hybrid regression model
# --------------------------------------------------------------------------- #
class QuantumRegression__gen399(nn.Module):
    """Hybrid regression model that optionally appends a quantum‑like feature
    extractor before the final linear head.
    """
    def __init__(self, num_features: int, use_quantum_features: bool = False):
        super().__init__()
        self.use_quantum = use_quantum_features
        self.base = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        if self.use_quantum:
            self.quantum_feat = RandomFourierFeatures(16, out_dim=32)
            self.head = nn.Linear(16 + 32, 1)
        else:
            self.head = nn.Linear(16, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        x = self.base(state_batch)
        if self.use_quantum:
            qfeat = self.quantum_feat(x)
            x = torch.cat([x, qfeat], dim=-1)
        return self.head(x).squeeze(-1)

__all__ = ["QuantumRegression__gen399", "RegressionDataset", "generate_superposition_data"]
