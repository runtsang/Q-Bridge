"""Enhanced convolution filter with depth‑wise separable structure, optional residual, and learnable threshold.

Designed as a drop‑in replacement for the original Conv filter, ConvPlus can be instantiated with the same signature. The implementation uses a depth‑wise convolution for each input channel, a point‑wise 1×1 convolution to fuse the channels, and an optional residual branch that adds the input back to the output. A learnable threshold is wrapped around the sigmoid activation so that the network can adjust the sensitivity of the filter during back‑propagation."""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["ConvPlus"]

class ConvPlus(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 kernel_size: int = 2,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 residual: bool = False,
                 bias: bool = True,
                 threshold_init: float = 0.0,
                 device: torch.device | None = None) -> None:
        super().__init__()
        self.residual = residual
        self.depthwise = nn.Conv2d(in_channels,
                                   in_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=in_channels,
                                   bias=bias,
                                   device=device)
        self.pointwise = nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=1,
                                   bias=bias,
                                   device=device)
        # learnable threshold
        self.threshold = nn.Parameter(torch.tensor(threshold_init, dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch, in_channels, H, W)
        Returns:
            Tensor of shape (batch, out_channels, H_out, W_out) after applying
            depthwise separable convolution, sigmoid with learned threshold,
            and optional residual addition.
        """
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = torch.sigmoid(out - self.threshold)
        if self.residual:
            if x.shape == out.shape:
                out = out + x
            else:
                proj = nn.Conv2d(self.depthwise.in_channels,
                                 self.pointwise.out_channels,
                                 kernel_size=1,
                                 bias=False).to(x.device)
                out = out + proj(x)
        return out
