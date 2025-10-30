"""Hybrid regression model combining classical convolution and fully connected layers.

This module defines a classical regression model that emulates the quantum behaviour
of the original `QuantumRegression.py` while adding a convolutional pre‑processor
and a fully‑connected linear head.  The design follows the *combination* scaling
paradigm: the classical backbone is extended with a conv filter that mimics the
quantum quanvolution, and the final regression head is a standard linear layer.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Data generation and dataset
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data.

    The data are drawn from a superposition distribution similar to the
    quantum seed.  The function returns a real‑valued feature matrix and a
    target vector.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that returns a 2‑D image for the convolutional filter."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        # reshape to a square image for the conv filter
        side = int(np.sqrt(self.features.shape[1]))
        return {
            "states": torch.tensor(self.features[index].reshape(1, side, side), dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Convolutional filter (classical stand‑in for quanvolution)
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """Simple 2‑D convolution followed by a sigmoid activation."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3])  # scalar per batch element

# --------------------------------------------------------------------------- #
# Fully‑connected (classical) layer
# --------------------------------------------------------------------------- #
class FullyConnectedLayer(nn.Module):
    """Linear layer that mimics the quantum fully‑connected layer."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(thetas)).mean(dim=0, keepdim=True)

# --------------------------------------------------------------------------- #
# Hybrid regression model
# --------------------------------------------------------------------------- #
class QModel(nn.Module):
    """Hybrid classical regression model."""
    def __init__(self, num_features: int, conv_kernel_size: int = 2):
        super().__init__()
        self.conv = ConvFilter(kernel_size=conv_kernel_size)
        # Conv output is scalar; feed into a small linear head
        self.fc = nn.Linear(1, 16)
        self.head = nn.Linear(16, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # state_batch shape: (batch, 1, side, side)
        conv_feat = self.conv(state_batch)  # (batch, 1)
        x = self.fc(conv_feat)
        return self.head(x).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
