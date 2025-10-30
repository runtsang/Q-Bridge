"""Hybrid multi‑channel convolution with optional depthwise reduction and learnable threshold."""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple, Optional, Iterable


class ConvEnhanced(nn.Module):
    """
    Drop‑in replacement for the original Conv filter.

    Parameters
    ----------
    in_channels : int
        Number of input feature maps.
    out_channels : int
        Number of output feature maps.
    kernel_size : int or tuple[int, int] = 2
        Size of the convolution kernel (square by default).
    depthwise : bool = False
        If True, use depthwise separable convolution (depthwise + point‑wise).
    threshold : float | None = None
        Learnable threshold applied after sigmoid. If None, a fixed value of 0.0 is used.
    bias : bool = True
        Whether to add a bias term in the convolution.
    """
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int] = 2,
        depthwise: bool = False,
        threshold: Optional[float] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.depthwise = depthwise

        if depthwise:
            # Depthwise conv: one filter per input channel
            self.depthwise_conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                groups=in_channels,
                bias=bias,
            )
            # Point‑wise conv to mix channels
            self.pointwise_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=bias,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
            )

        # Learnable threshold
        if threshold is None:
            self.threshold = nn.Parameter(torch.tensor(0.0))
        else:
            self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))

    def forward(self, data: torch.Tensor | Iterable) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch.Tensor or iterable of shape (B, C, H, W)

        Returns
        -------
        torch.Tensor
            Mean activation after sigmoid and threshold.
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        if data.ndim == 2:
            # Single sample, single channel
            data = data.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        elif data.ndim == 3 and data.shape[0] == 1:
            data = data.unsqueeze(0)  # (1,C,H,W)

        # Convolution
        if self.depthwise:
            x = self.depthwise_conv(data)
            x = self.pointwise_conv(x)
        else:
            x = self.conv(data)

        logits = x
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[1, 2, 3])  # mean per batch element

    def run(self, data) -> float:
        """Convenience wrapper that returns a single float."""
        return self.forward(data).mean().item()


__all__ = ["ConvEnhanced"]
