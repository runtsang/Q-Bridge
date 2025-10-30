"""Enhanced classical convolution filter with multi‑channel support and optional dropout.

This module provides a drop‑in replacement for the original Conv filter.  It
exposes a PyTorch ``nn.Module`` that supports multiple input/output channels,
arbitrary stride and padding, a thresholded sigmoid activation, and optional
dropout on the weights during training.  The public API mirrors the original
seed’s ``Conv`` factory: ``ConvFilter = Conv()``.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class ConvFilter(nn.Module):
    """Classical convolutional filter with trainable weights and optional dropout.

    Parameters
    ----------
    in_channels : int, default 1
        Number of input channels.
    out_channels : int, default 1
        Number of output channels (filters).
    kernel_size : int or tuple, default 3
        Size of the convolution kernel.
    stride : int or tuple, default 1
        Stride of the convolution.
    padding : int or tuple, default 0
        Zero‑padding added to both sides of the input.
    threshold : float, default 0.0
        Threshold applied after convolution before the sigmoid activation.
    dropout_rate : float, default 0.0
        Dropout probability applied to the filter weights during training.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        threshold: float = 0.0,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, weight‑dropout, and sigmoid activation."""
        weight = self.dropout(self.conv.weight)
        bias = self.conv.bias
        conv_out = F.conv2d(
            x,
            weight,
            bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
        )
        return torch.sigmoid(conv_out - self.threshold)

    def run(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Convenience wrapper that accepts a NumPy array and returns a torch tensor."""
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=torch.float32)
        return self.forward(data)


def Conv() -> ConvFilter:
    """Factory that returns a ``ConvFilter`` instance with default parameters."""
    return ConvFilter()


__all__ = ["ConvFilter", "Conv"]
