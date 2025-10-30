"""Hybrid classical convolutional filter with adaptive threshold and batch support.

This module extends the original Conv filter by adding a learnable threshold,
supporting batched inputs, and providing a drop‑in PyTorch interface.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np

class ConvEnhanced(nn.Module):
    """
    A hybrid convolutional filter that can be used as a drop‑in replacement
    for the original Conv class. It accepts 2‑D or 3‑D tensors (batch, H, W)
    and performs a 2‑D convolution followed by a sigmoid activation that is
    controlled by a learnable threshold.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        init_threshold: float = 0.0,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Learnable threshold parameter
        self.threshold = nn.Parameter(
            torch.full((1,), init_threshold, device=device)
        )
        # 2‑D convolution layer
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that accepts a tensor of shape
        (B, 1, H, W) or (1, H, W) and returns the mean activation.
        """
        # Ensure input has 4 dimensions
        if x.ndim == 3:
            x = x.unsqueeze(0)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

    # Compatibility with the original API
    def run(self, data):
        """
        Accepts a 2‑D array or list and returns a scalar float.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return self.forward(tensor).item()

__all__ = ["ConvEnhanced"]
