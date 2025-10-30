"""Enhanced classical convolutional filter with multi‑channel, depth‑wise separable, and optional attention.

The ConvGen class can be used as a drop‑in replacement for the original Conv() factory.
It supports an optional depth‑wise separable convolution and a simple channel‑wise attention
mechanism.  The API remains the same: a Conv() function returns an instance of ConvGen,
and a run() method accepts a 2‑D or 3‑D NumPy array and returns a scalar score.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Any

class ConvGen(nn.Module):
    """
    Classical depth‑wise separable convolution with optional self‑attention.
    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel.
    num_channels : int, default 1
        Number of input/output channels.
    depthwise : bool, default False
        If True use a depth‑wise + point‑wise (1×1) convolution.
    self_attention : bool, default False
        If True add a channel‑wise attention block (Squeeze‑and‑Excitation).
    threshold : float, default 0.0
        Threshold used by the original ConvFilter for binarising input.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        num_channels: int = 1,
        depthwise: bool = False,
        self_attention: bool = False,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.threshold = threshold
        self.depthwise = depthwise
        self.self_attention = self_attention

        # Convolution layers
        if depthwise:
            self.depthwise_conv = nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=kernel_size,
                bias=True,
                groups=num_channels,
            )
            self.pointwise_conv = nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=1,
                bias=True,
            )
        else:
            self.depthwise_conv = nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=kernel_size,
                bias=True,
            )
            self.pointwise_conv = None

        # Optional self‑attention (Squeeze‑and‑Excitation)
        if self_attention:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(num_channels, max(1, num_channels // 8), 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(1, num_channels // 8), num_channels, 1),
                nn.Sigmoid(),
            )
        else:
            self.attention = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the convolutional filter.
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, H, W).
        Returns
        -------
        torch.Tensor
            Output tensor after optional depth‑wise and attention layers.
        """
        out = self.depthwise_conv(x)
        if self.pointwise_conv is not None:
            out = self.pointwise_conv(out)
        if self.attention is not None:
            w = self.attention(out)
            out = out * w
        return out

    def run(self, data: Any) -> float:
        """
        Convenience method that accepts a NumPy array and returns the mean activation
        value, mimicking the behaviour of the original ConvFilter.
        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) or
            3‑D array of shape (num_channels, kernel_size, kernel_size).
        Returns
        -------
        float
            Mean activation after convolution and optional attention.
        """
        if isinstance(data, np.ndarray):
            data_np = data
        else:
            raise TypeError("Input data must be a NumPy array")
        # Convert to torch tensor
        if data_np.ndim == 2:
            # Single channel
            tensor = torch.from_numpy(data_np).float().unsqueeze(0).unsqueeze(0)
        elif data_np.ndim == 3:
            # Multi‑channel
            tensor = torch.from_numpy(data_np).float().unsqueeze(0)
        else:
            raise ValueError("Unsupported input shape")
        with torch.no_grad():
            out = self.forward(tensor)
        return out.mean().item()

def Conv() -> ConvGen:
    """
    Factory function that returns a default ConvGen instance,
    keeping backward compatibility with the original Conv() callable.
    """
    return ConvGen()

__all__ = ["Conv"]
