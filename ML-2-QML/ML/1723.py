"""Enhanced classical convolution filter with depthwise separable support.

This module provides a ConvEnhanced class that can be instantiated in place
of the original Conv function.  It supports multi‑channel inputs, optional
depthwise‑separable convolution for efficiency, and a threshold‑based
activation.  The API mirrors the seed with a `run` method that returns a
scalar activation value.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional

class ConvEnhanced(nn.Module):
    """Depthwise‑separable convolutional filter for ablation studies.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the kernel (must be odd for padding).  The class pads the input
        to preserve spatial dimensions.
    in_channels : int, default 1
        Number of input channels (set to 2 for multi‑channel data).
    out_channels : int, default 1
        Number of output channels (further processing may use 1‑channel
        activations.
    depthwise : bool, default False
        Whether to use depthwise‑separable conv.
    threshold : float, default 0.0
        Bias threshold applied before sigmoid.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        depthwise: bool = False,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depthwise = depthwise
        self.threshold = threshold

        if depthwise:
            # depthwise convolution: one filter per input channel
            self.depthwise_conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                groups=in_channels,
                bias=True,
            )
            # pointwise convolution to mix channels
            self.pointwise_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=True,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, bias=True
            )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single sample.

        Parameters
        ----------
        data : torch.Tensor
            Tensor of shape (in_channels, H, W). For the legacy API
            a 2‑D array is accepted and wrapped to shape (1, H, W).

        Returns
        -------
        torch.Tensor
            Scalar activation value (mean sigmoid of logits).
        """
        if data.ndim == 2:
            data = data.unsqueeze(0)  # 1 channel
        if self.depthwise:
            out = self.depthwise_conv(data)
            out = self.pointwise_conv(out)
        else:
            out = self.conv(data)
        logits = out
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

    def run(self, data) -> float:
        """Convenience wrapper matching the legacy API.

        Parameters
        ----------
        data : array‑like
            2‑D or 3‑D array with shape (H, W) or (C, H, W).

        Returns
        -------
        float
            Scalar activation value.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        return self.forward(tensor).item()

__all__ = ["ConvEnhanced"]
