"""Hybrid classical regression model with convolutional feature extractor."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data.
    Features are drawn uniformly in [-1, 1] and the target is a smooth
    sinusoidal function of the feature sum, mimicking a quantum superposition.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset that reshapes the 1‑D feature vector into a 2‑D grid
    for convolutional processing. The target remains a scalar.
    """
    def __init__(self, samples: int, num_features: int, kernel_size: int = 2):
        self.kernel_size = kernel_size
        self.features, self.labels = generate_superposition_data(num_features, samples)
        # reshape to (kernel_size, kernel_size)
        self.features = self.features.reshape(-1, kernel_size, kernel_size)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ConvFilter(nn.Module):
    """
    Simple 2‑D convolutional filter that emulates a quantum filter.
    It returns a global average of the sigmoid‑activated logits.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, H, W)
        x = x.unsqueeze(1)  # (batch, 1, H, W)
        logits = self.conv(x)  # (batch, 1, H-k+1, W-k+1)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3])  # global average pooling

class HybridRegression(nn.Module):
    """Hybrid classical regression model with a convolutional front‑end."""
    def __init__(self, num_features: int, kernel_size: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = ConvFilter(kernel_size=kernel_size)
        # after global average, we have a single feature per sample
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # state_batch: (batch, H, W)
        features = self.conv(state_batch)  # (batch, 1)
        return self.net(features).squeeze(-1)

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
