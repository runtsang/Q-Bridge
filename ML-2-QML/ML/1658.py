"""HybridConvLayer: classical convolutional backbone with optional depthwise separable conv.

This module extends the original single-filter design by providing a small
convolutional network that can be used as a drop‑in replacement for the
quantum filter.  It supports batch‑normalisation, dropout, and a
parameter‑efficient depth‑wise separable option.  The forward method
returns a feature map that can be fed into a downstream classifier or
into a quantum filter for hybrid experimentation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

class HybridConvLayer(nn.Module):
    """A lightweight, fully‑classical convolutional module."""
    def __init__(
        self,
        kernel_size: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        depthwise: bool = False,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.depthwise = depthwise

        if depthwise:
            self.depth_conv = nn.Conv2d(
                in_channels, in_channels, kernel_size,
                stride=stride, padding=padding, groups=in_channels, bias=bias
            )
            self.pointwise_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                stride=1, padding=0, bias=bias
            )
            conv = self.depth_conv
        else:
            conv = nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=bias
            )

        self.conv = conv
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid layer."""
        if self.depthwise:
            x = self.depth_conv(x)
            x = self.pointwise_conv(x)
        else:
            x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(x)

        return F.relu(x)

__all__ = ["HybridConvLayer"]
