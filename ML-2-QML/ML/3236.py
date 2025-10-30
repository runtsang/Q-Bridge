"""Hybrid convolutional filter that unifies classical and quantum-inspired designs.

The module implements a 2×2 kernel with optional stride and threshold gating,
producing four output channels. It can be used as a drop‑in replacement for
the original ``Conv`` class while extending its functionality with a richer
feature map.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["HybridConvFilter"]


class HybridConvFilter(nn.Module):
    """
    Classical convolutional filter with 4 output channels and optional
    threshold gating.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = 2,
        threshold: float = 0.0,
        use_threshold: bool = True,
        num_filters: int = 4,
    ) -> None:
        """
        Args:
            kernel_size: Size of the convolution kernel.
            stride: Stride of the convolution.
            threshold: Threshold value for sigmoid gating.
            use_threshold: If True, apply sigmoid(logits - threshold).
            num_filters: Number of output channels (default 4).
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.threshold = threshold
        self.use_threshold = use_threshold
        self.conv = nn.Conv2d(
            1,
            num_filters,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
        )
        # Initialise weights to emulate a random quantum kernel
        nn.init.uniform_(self.conv.weight, -1.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution and optional threshold gating.

        Args:
            x: Input tensor of shape (batch, 1, H, W).

        Returns:
            Flattened feature tensor of shape (batch, num_patches * num_filters).
        """
        logits = self.conv(x)
        if self.use_threshold:
            logits = torch.sigmoid(logits - self.threshold)
        # Flatten the feature map for compatibility with linear heads
        return logits.view(x.size(0), -1)
