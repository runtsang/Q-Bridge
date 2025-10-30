"""Hybrid Quanvolution regressor using classical convolution and a regression head.

This module combines the classical quanvolution idea with a regression head inspired by QuantumRegression.
"""

import torch
import torch.nn as nn
import numpy as np

class QuanvolutionRegressor(nn.Module):
    """Classical quanvolution filter followed by a regression head."""
    def __init__(self, patch_size: int = 2, stride: int = 2, out_channels: int = 4):
        super().__init__()
        # 2×2 patches, stride 2 → 14×14 feature map for 28×28 input
        self.conv = nn.Conv2d(1, out_channels, kernel_size=patch_size, stride=stride)
        self.linear = nn.Linear(out_channels * 14 * 14, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        features = features.view(x.size(0), -1)
        out = self.linear(features)
        return out.squeeze(-1)

def generate_image_regression_data(num_samples: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate random 28×28 grayscale images and targets as the sum of pixel values."""
    rng = np.random.default_rng(seed)
    images = rng.uniform(0.0, 1.0, size=(num_samples, 1, 28, 28)).astype(np.float32)
    targets = images.reshape(num_samples, -1).sum(axis=1).astype(np.float32)
    return images, targets

__all__ = ["QuanvolutionRegressor", "generate_image_regression_data"]
