"""Hybrid classical regression model inspired by quantum random layers and sampler networks.

This module defines:
- generate_superposition_data: reproduces the dataset from the original seed.
- RegressionDataset: PyTorch Dataset wrapping the data.
- SamplerQNN: a lightweight neural network that emulates a quantum sampler.
- HybridRegressionModel: a two‑branch network combining a classical encoder, a quantum‑inspired
  random layer, and the sampler network, followed by a linear head.

The design facilitates direct comparison with the quantum counterpart while remaining fully
classical and GPU‑friendly.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate dataset with features x and labels y = sin(sum(x)) + 0.1*cos(2*sum(x))."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset wrapping the superposition data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class SamplerQNN(nn.Module):
    """Classical surrogate for a quantum sampler network."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)

class QuantumFeatureExtractor(nn.Module):
    """A classical layer mimicking a quantum random layer."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # Random matrix is fixed (non‑trainable) to emulate a random feature map.
        self.register_buffer("random_matrix", torch.randn(out_dim, in_dim))
        self.linear = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.matmul(x, self.random_matrix.t())
        x = torch.tanh(x)
        return self.linear(x)

class HybridRegressionModel(nn.Module):
    """Two‑branch classical model combining encoder, quantum‑inspired layer, and sampler."""
    def __init__(self, num_features: int, hidden_dim: int = 32):
        super().__init__()
        self.encoder = nn.Linear(num_features, hidden_dim)
        self.quantum_layer = QuantumFeatureExtractor(hidden_dim, hidden_dim)
        self.sampler = SamplerQNN()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.quantum_layer(x)
        x = self.sampler(x)
        return self.head(x).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data", "SamplerQNN"]
