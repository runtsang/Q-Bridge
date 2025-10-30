"""Hybrid convolutional layer with trainable weights.

The original drop‑in filter is replaced by a fully‑trainable
`ConvLayer` that supports multi‑channel inputs, stride, padding and
bias.  A convenience factory `Conv()` keeps the original callable
interface, allowing existing pipelines to use the new class without
modification.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Conv"]

class ConvLayer(nn.Module):
    """Trainable convolutional filter.

    Args:
        kernel_size: Size of the square kernel.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride of the convolution.
        padding: Zero‑padding added to both sides.
        bias: Whether to add a learnable bias.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size,
                dtype=torch.float32,
            )
        )
        self.bias = (
            nn.Parameter(torch.empty(out_channels, dtype=torch.float32))
            if bias
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and return sigmoid activation."""
        y = F.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        return torch.sigmoid(y)

    def run(self, data) -> torch.Tensor:
        """Convenience wrapper that accepts raw data.

        Returns the mean activation over the batch.
        """
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=torch.float32)
        # Ensure shape: (B, C, H, W)
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)
        elif data.ndim == 3:
            data = data.unsqueeze(0)
        return self.forward(data).mean()

def Conv(
    kernel_size: int = 2,
    in_channels: int = 1,
    out_channels: int = 1,
) -> ConvLayer:
    """Return a callable object that emulates the quantum filter
    with PyTorch ops."""
    return ConvLayer(kernel_size, in_channels, out_channels)
