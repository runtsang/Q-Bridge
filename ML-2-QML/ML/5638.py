"""Hybrid classical regression with convolutional preprocessing."""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data with cosine/sine patterns."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that reshapes 1‑D features into a 2‑D grid for convolution."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)
        # Ensure the feature dimension is a perfect square for 2‑D reshaping.
        sqrt_dim = int(math.isqrt(num_features))
        if sqrt_dim ** 2!= num_features:
            raise ValueError(
                f"num_features ({num_features}) must be a perfect square for 2‑D conv."
            )
        self.sqrt_dim = sqrt_dim

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel(nn.Module):
    """Hybrid classical regression model with a 2‑D convolution followed by fully‑connected layers."""

    def __init__(self, num_features: int, kernel_size: int = 2, out_channels: int = 8):
        super().__init__()
        sqrt_dim = int(math.isqrt(num_features))
        if sqrt_dim ** 2!= num_features:
            raise ValueError(
                f"num_features ({num_features}) must be a perfect square for 2‑D conv."
            )
        self.sq_dim = sqrt_dim

        # Convolutional front‑end
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )
        # Compute the spatial size after convolution
        conv_out = self.sq_dim - kernel_size + 1
        conv_features = out_channels * conv_out * conv_out

        # Fully‑connected back‑end
        self.fc = nn.Sequential(
            nn.Linear(conv_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Reshape to (batch, 1, H, W)
        batch = state_batch.shape[0]
        reshaped = state_batch.view(batch, 1, self.sq_dim, self.sq_dim)
        conv_out = self.conv(reshaped)
        flattened = conv_out.reshape(batch, -1)
        return self.fc(flattened).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
