"""Hybrid classical estimator combining a quanvolution filter and a fully connected layer.

The model mirrors the EstimatorQNN example but extends it with a
quantum-inspired convolution (QuanvolutionFilter) and a flexible
fully connected head.  It is fully compatible with the original
anchor path and can be used interchangeably with the original
EstimatorQNN class.

The architecture:
    1. 2×2 patches are extracted via a 2‑D convolution.
    2. The resulting feature map is flattened.
    3. A linear layer maps the features to a single output.

The module exposes ``EstimatorQNN`` as a factory returning an
``nn.Module`` instance.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """
    Classical convolutional filter that emulates the behaviour of a
    quantum 2×2 patch kernel.  It keeps the interface of the original
    quanvolution example while being lightweight and fully differentiable.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.conv(x).view(x.size(0), -1)


class FullyConnectedLayer(nn.Module):
    """
    A minimal fully connected layer that can be used as the head of a
    hybrid architecture.  It mimics the behaviour of the FCL example
    while remaining fully classical.
    """

    def __init__(self, in_features: int, out_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear(x)


class HybridEstimator(nn.Module):
    """
    Combines the quanvolution filter and a fully connected head.
    The forward pass returns a single scalar output, suitable for
    regression tasks.
    """

    def __init__(self, in_channels: int = 1, n_patches: int = 14 * 14, out_features: int = 1) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels)
        # 4 output channels from the convolution and 14×14 patches
        self.fc = FullyConnectedLayer(in_features=4 * n_patches, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        return self.fc(features)


class EstimatorQNN(nn.Module):
    """
    Public API compatible with the original EstimatorQNN module.
    It simply wraps the HybridEstimator.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = HybridEstimator()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)


def EstimatorQNN() -> EstimatorQNN:
    """Return a hybrid estimator model."""
    return EstimatorQNN()


__all__ = ["EstimatorQNN", "HybridEstimator", "QuanvolutionFilter", "FullyConnectedLayer"]
