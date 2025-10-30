"""Classical regression model mirroring the quantum example.

Provides a drop‑in replacement that uses a simple feed‑forward network and a classical convolution filter.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
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


class ConvFilter(nn.Module):
    """Simple 2‑D convolution filter that mimics a quantum quanvolution."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, H, W)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3])


class QuantumRegressionModel(nn.Module):
    """Classical regression network that can be replaced by the quantum version."""
    def __init__(self, num_features: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch).squeeze(-1)


__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data", "ConvFilter"]
