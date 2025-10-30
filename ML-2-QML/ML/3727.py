"""Combined classical kernel with convolutional feature extraction."""

from __future__ import annotations

from typing import Sequence, Iterable

import numpy as np
import torch
from torch import nn


class ConvFeatureExtractor(nn.Module):
    """Two‑dimensional convolutional filter that emulates a quanvolution layer.

    The module applies a single learnable 2×2 kernel followed by a sigmoid
    non‑linearity and returns the mean activation as a scalar feature.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data shape: (batch, height, width) or (height, width)
        if data.dim() == 2:
            data = data.unsqueeze(0).unsqueeze(0)
        elif data.dim() == 3:
            data = data.unsqueeze(1)
        else:
            raise ValueError("Input tensor must be 2‑ or 3‑D.")
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean([-2, -1])  # mean over spatial dimensions


class RBFKernel(nn.Module):
    """Radial‑basis function kernel that operates on feature vectors."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (n, d), y: (m, d)
        diff = x.unsqueeze(1) - y.unsqueeze(0)          # (n, m, d)
        sq_norm = (diff * diff).sum(dim=-1)              # (n, m)
        return torch.exp(-self.gamma * sq_norm)


def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  gamma: float = 1.0,
                  device: torch.device | None = None) -> np.ndarray:
    """Compute the Gram matrix between two datasets with optional GPU support."""
    if device is None:
        device = torch.device("cpu")
    kernel = RBFKernel(gamma).to(device)
    feats_a = torch.cat([ConvFeatureExtractor()(x.to(device)) for x in a], dim=0)
    feats_b = torch.cat([ConvFeatureExtractor()(x.to(device)) for x in b], dim=0)
    mat = kernel(feats_a, feats_b)
    return mat.cpu().numpy()


__all__ = ["ConvFeatureExtractor", "RBFKernel", "kernel_matrix"]
