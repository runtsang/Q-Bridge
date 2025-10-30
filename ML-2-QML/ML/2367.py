"""Hybrid quanvolution regression model combining classical convolutional filtering with a regression head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridQuanvolutionRegression(nn.Module):
    """Classical hybrid model that applies a 2x2 convolutional filter to image patches and
    then predicts a scalar regression target using a linear head.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 4,
        patch_size: int = 2,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.patch_conv = nn.Conv2d(
            in_channels, out_features, kernel_size=patch_size, stride=stride
        )
        # Compute number of patches per dimension
        self.num_patches = ((28 - patch_size) // stride + 1) ** 2
        self.regressor = nn.Sequential(
            nn.Linear(out_features * self.num_patches, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, height, width)
        conv_out = self.patch_conv(x)
        flat = conv_out.view(conv_out.size(0), -1)
        return self.regressor(flat).squeeze(-1)


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data using a superposition-inspired target function."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapper around the synthetic data."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


__all__ = ["HybridQuanvolutionRegression", "RegressionDataset", "generate_superposition_data"]
