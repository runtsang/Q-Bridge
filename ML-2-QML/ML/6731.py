"""Hybrid classical kernel combining RBF and quanvolution features."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RBFKernel(nn.Module):
    """Radial basis function kernel with configurable gamma."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class QuanvolutionFilter(nn.Module):
    """Convolutional filter inspired by the quanvolution concept."""

    def __init__(self, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


class HybridKernel(nn.Module):
    """
    Hybrid kernel that combines a classical RBF kernel with quanvolution-based features.
    The kernel value between two samples is the sum of the RBF similarity and the
    dot product of their quanvolution feature vectors.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.rbf = RBFKernel(gamma)
        self.quanvolution = QuanvolutionFilter(out_channels, kernel_size, stride)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # RBF component on flattened vectors
        x_flat, y_flat = x.view(x.size(0), -1), y.view(y.size(0), -1)
        rbf_val = self.rbf(x_flat, y_flat)

        # Quanvolution component
        qx = self.quanvolution(x)
        qy = self.quanvolution(y)
        quanv_val = torch.sum(qx * qy, dim=-1, keepdim=True)

        return rbf_val + quanv_val

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute the Gram matrix for two collections of samples."""
        return np.array(
            [[self.forward(x, y).item() for y in b] for x in a]
        )


__all__ = ["RBFKernel", "QuanvolutionFilter", "HybridKernel"]
