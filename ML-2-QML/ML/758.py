"""Hybrid depth‑wise separable convolution module with optional batch‑norm.

This module implements a drop‑in replacement for the original Conv filter.
It supports a depth‑wise separable convolution (optionally followed by a point‑wise
convolution) and an optional batch‑norm layer. The output is a single scalar
obtained by mean pooling over the feature map.

The module can be used in a standard PyTorch training loop and is fully
differentiable.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["ConvEnhanced"]


class ConvEnhanced(nn.Module):
    """
    Depth‑wise separable convolution filter with optional batch‑norm.

    Parameters
    ----------
    kernel_size : int
        Size of the square kernel.
    threshold : float
        Threshold applied to the activation before mean pooling.
    depthwise : bool
        If True, use depth‑wise separable convolution.
    use_batchnorm : bool
        If True, add a batch‑norm layer after the convolution.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        depthwise: bool = False,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.depthwise = depthwise
        self.use_batchnorm = use_batchnorm

        in_channels = 1
        out_channels = 1

        if depthwise:
            # depth‑wise conv (groups=in_channels)
            self.depthwise_conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                bias=True,
                groups=in_channels,
            )
            self.pointwise_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=True
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=True)

        if use_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Scalar output of shape (batch,).
        """
        if self.depthwise:
            x = self.depthwise_conv(x)
            x = self.pointwise_conv(x)
        else:
            x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x)

        # Apply threshold
        x = torch.sigmoid(x - self.threshold)

        # Global mean pooling to scalar
        return x.mean(dim=(1, 2, 3))

    def run(self, data: torch.Tensor) -> float:
        """
        Convenience method that accepts a single 2‑D array and returns a scalar.

        Parameters
        ----------
        data : torch.Tensor
            2‑D tensor of shape (H, W).

        Returns
        -------
        float
            Mean activation after thresholding.
        """
        self.eval()
        with torch.no_grad():
            tensor = data.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            return self.forward(tensor).item()
