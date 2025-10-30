"""Enhanced convolutional filter with depthwise separable layers and adaptive thresholding."""

from __future__ import annotations

import torch
from torch import nn


def ConvGen130() -> nn.Module:
    """Return a ConvGen130 module that can replace the original Conv.

    The module supports an adaptive kernel size (1–4) and learns a threshold
    that is applied before the sigmoid activation.  The design follows the
    depthwise–separable pattern commonly used in MobileNet‑style
    classifiers, which keeps the number of trainable parameters small
    while still providing expressive power for feature extraction.
    """

    class ConvGen130(nn.Module):
        def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            kernel_size: int = 2,
            depthwise: bool = True,
            threshold_init: float = 0.0,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.depthwise = depthwise
            self.threshold = nn.Parameter(
                torch.tensor(threshold_init, dtype=torch.float32)
            )

            # Depth‑wise convolution
            self.depthwise_conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=in_channels,
                bias=True,
            )

            # Point‑wise convolution
            self.pointwise_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=True,
            )

            self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply depthwise‑separable convolution then sigmoid‑threshold.

            Args:
                x: Tensor of shape (N, C_in, H, W).

            Returns:
                Tensor: mean activation after sigmoid threshold.
            """
            x = self.depthwise_conv(x)
            x = self.pointwise_conv(x)
            if self.dropout is not None:
                x = self.dropout(x)

            # Apply learnable threshold before sigmoid
            x = torch.sigmoid(x - self.threshold)
            return x.mean()

    return ConvGen130()
