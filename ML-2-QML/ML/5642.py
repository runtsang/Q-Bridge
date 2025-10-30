"""Hybrid classical regression model combining superposition data generation
and a 2‑D convolutional feature extractor."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a superposition‑like dataset and reshape features into a square image."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    side = int(np.sqrt(num_features))
    assert side * side == num_features, "num_features must be a perfect square for 2‑D reshaping"
    x = x.reshape(samples, 1, side, side)  # (batch, channel, height, width)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset yielding 2‑D feature maps and scalar targets."""
    def __init__(self, samples: int, num_features: int):
        super().__init__()
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """Convolutional regression head that mimics a quanvolution filter."""
    def __init__(self, num_features: int):
        super().__init__()
        side = int(np.sqrt(num_features))
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        conv_out = ((side - 2) // 2 + 1) * 4  # feature count after conv
        self.head = nn.Sequential(
            nn.Linear(conv_out, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        x = self.conv(state_batch)
        x = x.view(state_batch.size(0), -1)
        return self.head(x).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
