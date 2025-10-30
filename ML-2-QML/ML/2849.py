"""Hybrid classical convolutional filter with learnable weights and thresholding.

This module combines ideas from the original Conv.py and Quanvolution.py
to provide a versatile filter that can be used as a drop‑in replacement
for the quantum filter.  The filter supports multiple output channels,
learnable threshold, and a linear classifier head.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class HybridConvFilter(nn.Module):
    """Classical convolutional filter with learnable weights.

    The filter implements a 2×2 kernel with a learnable threshold
    applied to a sigmoid activation.  It outputs `out_channels`
    feature maps, matching the channel count of the Quanvolution
    filter, and can be used as a drop‑in replacement.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        out_channels: int = 4,
        stride: int = 2,
        threshold: float | torch.Tensor = 0.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
        )
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a feature map after sigmoid thresholding.

        Parameters
        ----------
        x : torch.Tensor
            Input image of shape ``(B, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Feature map of shape ``(B, out_channels, H', W')`` where
            ``H' = (H - kernel_size) // stride + 1``.
        """
        logits = self.conv(x)
        # Apply a learnable threshold before sigmoid
        activations = torch.sigmoid(logits - self.threshold)
        return activations


class HybridConvClassifier(nn.Module):
    """Hybrid classifier that stacks the filter and a linear head."""

    def __init__(
        self,
        num_classes: int = 10,
        kernel_size: int = 2,
        out_channels: int = 4,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.qfilter = HybridConvFilter(kernel_size, out_channels, stride)
        # Compute feature map size after convolution
        dummy = torch.zeros(1, 1, 28, 28)
        fmaps = self.qfilter(dummy)
        feat_dim = fmaps.view(1, -1).size(1)
        self.linear = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features.view(features.size(0), -1))
        return F.log_softmax(logits, dim=-1)


def Conv() -> HybridConvFilter:
    """Return a ready‑to‑use filter instance."""
    return HybridConvFilter()


__all__ = ["HybridConvFilter", "HybridConvClassifier", "Conv"]
