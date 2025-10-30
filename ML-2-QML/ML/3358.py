"""Hybrid classical convolution module inspired by Conv.py and QuanvolutionFilter.

The class exposes a drop‑in replacement for the original Conv filter, but
extends it with a learnable 2×2 kernel, optional stride, and a
threshold‑based binarization step.  The module can be used either
stand‑alone or as a building block in a larger CNN.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class HybridConv(nn.Module):
    """
    Classical 2×2 convolution with optional thresholding.

    Parameters
    ----------
    in_channels : int
        Number of input channels. Default is 1 (grayscale).
    out_channels : int
        Number of output channels. Default is 4, matching the
        quantum filter's output dimensionality.
    kernel_size : int
        Size of the convolutional kernel. Fixed to 2 for compatibility
        with the quantum counterpart.
    stride : int
        Stride of the convolution. Default is 2 to downsample the
        feature map and match the patch extraction in the quantum
        implementation.
    threshold : float
        Value used to binarize the input patch before convolution.
        Pixels > threshold are set to 1, otherwise 0.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Extracted features of shape (N, out_channels, H', W').
        """
        # Binarize the input patch based on the threshold
        binarized = (x > self.threshold).float()
        return self.conv(binarized)

    def run(self, data: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper that accepts a single image patch
        and returns the mean activation across channels.

        Parameters
        ----------
        data : torch.Tensor
            Patch of shape (C, H, W) with H=W=kernel_size.

        Returns
        -------
        torch.Tensor
            Mean activation value (scalar).
        """
        with torch.no_grad():
            # Add batch dimension
            patch = data.unsqueeze(0)
            out = self.forward(patch)
            return out.mean().item()


__all__ = ["HybridConv"]
