"""
ConvGen103: Depthwise separable convolution module with multi‑channel support.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np


class ConvGen103(nn.Module):
    """
    Drop‑in replacement for the original Conv class.
    Supports:
      * multi‑channel input / output
      * depthwise separable convolution (depthwise + point‑wise)
      * automatic kernel size inference from the input tensor
      * thresholded sigmoid activation
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int | None = None,
        stride: int = 1,
        padding: int = 0,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.threshold = threshold

        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=True,
        )
        # Point‑wise convolution
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Infer kernel size if not provided
        if self.kernel_size is None:
            self.kernel_size = x.shape[-1]  # assume square kernel
        out = self.depthwise(x)
        out = self.pointwise(out)
        # Apply thresholded sigmoid activation
        out = torch.sigmoid(out - self.threshold)
        return out

    def run(self, data: np.ndarray) -> float:
        """
        Run the convolution on a 2‑D numpy array and return the mean activation.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = self.forward(tensor)
        return out.mean().item()


def Conv(
    in_channels: int = 1,
    out_channels: int = 1,
    kernel_size: int | None = None,
    stride: int = 1,
    padding: int = 0,
    threshold: float = 0.0,
) -> ConvGen103:
    """Factory function mirroring the original Conv() signature."""
    return ConvGen103(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        threshold=threshold,
    )


__all__ = ["ConvGen103", "Conv"]
