"""Hybrid regression module combining QCNN-inspired classical layers with a quantum backend.

This module defines:
- generate_superposition_data: synthetic regression data generator.
- RegressionDataset: PyTorch Dataset wrapping the data.
- HybridRegression: a classical neural network that emulates the QCNN architecture
  and can be trained using standard PyTorch optimizers.

The architecture mirrors the QCNN from the reference while adding an input layer
to adapt arbitrary feature dimensionality.  It is fully compatible with the
anchor `QuantumRegression.py` dataset utilities.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for regression.
    The data mimics a superposition of basis states with a sinusoidal target.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    PyTorch Dataset wrapping the synthetic regression data.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridRegression(nn.Module):
    """
    Classical regression model that emulates a QCNN architecture.
    The network consists of:
    - An input linear layer mapping arbitrary feature dimension to 8.
    - A series of fully‑connected layers that mimic QCNN's convolution and pooling steps.
    - A final linear head producing a scalar regression output.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.input_layer = nn.Linear(num_features, 8)
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.input_layer(state_batch)
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x)).squeeze(-1)

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
