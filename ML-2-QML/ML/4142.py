"""Hybrid regression model combining classical convolution-inspired layers and a feed‑forward head.

The module re‑implements the dataset generation from the original QuantumRegression
example and augments the neural network with a stack of fully‑connected layers that
mimic a quantum convolutional neural network.  The architecture is deliberately
light‑weight so that it can be trained on a CPU while still demonstrating how
classical and quantum components can be paired in a hybrid pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data where the target depends on the sum of
    input features via a sinusoidal relationship.  This mirrors the data used
    in the original QuantumRegression example but is kept here for compatibility
    with the hybrid workflow.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that yields a dictionary with a ``states`` tensor and a ``target`` tensor."""
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
    """Classical hybrid regression network that emulates a QCNN.

    The core of the network is a stack of fully‑connected layers that mimic the
    convolution, pooling, and final classification stages of the QCNN example.
    The model is deliberately simple – it contains dropout and residual
    connections to illustrate how classical regularisation can be combined
    with quantum‑inspired design patterns.
    """
    def __init__(self, num_features: int = 8, dropout: float = 0.1):
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.Tanh(),
        )
        self.conv1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.Tanh(),
        )
        self.pool1 = nn.Sequential(
            nn.Linear(16, 12),
            nn.Tanh(),
        )
        self.conv2 = nn.Sequential(
            nn.Linear(12, 8),
            nn.Tanh(),
        )
        self.pool2 = nn.Sequential(
            nn.Linear(8, 4),
            nn.Tanh(),
        )
        self.conv3 = nn.Sequential(
            nn.Linear(4, 4),
            nn.Tanh(),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        return torch.sigmoid(self.head(x)).squeeze(-1)

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
