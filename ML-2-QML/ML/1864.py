"""Classical convolutional filter with multi‑channel support and gating.

Features:
- Multi‑channel input and output.
- Learnable kernel weights via Conv2d.
- Sigmoid gating controlled by a threshold.
- Compatible with torch.nn.Sequential and transfer to GPU.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Any

class Conv(nn.Module):
    """Multi‑channel convolution with sigmoid gating."""
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 2,
        stride: int = 1,
        padding: int = 0,
        threshold: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.threshold = threshold
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and optional sigmoid gating."""
        conv_out = self.conv(x)
        if self.threshold!= 0.0:
            gate = torch.sigmoid(conv_out - self.threshold)
            conv_out = conv_out * gate
        return conv_out

    def to(self, *args: Any, **kwargs: Any) -> "Conv":
        """Transfer module to a device."""
        return super().to(*args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"Conv(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding})"
        )

__all__ = ["Conv"]
