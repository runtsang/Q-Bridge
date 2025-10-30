"""ConvGen275: Classical depth‑wise separable convolution with learnable threshold."""

from __future__ import annotations

import torch
from torch import nn

class ConvGen275(nn.Module):
    """
    Drop‑in replacement for the original Conv filter.
    Supports multi‑channel input, batch processing, and a learnable threshold.
    Implements a depth‑wise separable convolution: first a depth‑wise conv per channel,
    then a 1×1 point‑wise conv to mix channels.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        threshold_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # depth‑wise convolution: groups=in_channels
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

        # point‑wise convolution: 1×1 kernel
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )

        # learnable threshold
        self.threshold = nn.Parameter(torch.tensor(threshold_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, in_channels, H, W).

        Returns:
            Tensor of shape (batch, out_channels, H_out, W_out) after sigmoid
            activation with learnable threshold.
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        # apply sigmoid with threshold
        x = torch.sigmoid(x - self.threshold)
        return x

    def run(self, data) -> float:
        """
        Run the filter on raw data and return the mean activation.

        Args:
            data: NumPy array or torch tensor of shape
                  (batch, in_channels, H, W) or (in_channels, H, W).

        Returns:
            Mean activation as a Python float.
        """
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=torch.float32)
        if data.ndim == 3:
            # add batch dimension
            data = data.unsqueeze(0)
        out = self.forward(data)
        return out.mean().item()

__all__ = ["ConvGen275"]
