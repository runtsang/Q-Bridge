"""Enhanced classical convolutional filter with multi‑channel support and learnable threshold.

The original Conv() helper returned a single‑channel, fixed‑threshold filter.  In many image‑classification tasks the filter must operate over several input channels (e.g. RGB), and the threshold can be tuned during training.  This module exposes a `ConvFilter` that
* accepts an arbitrary number of input channels,
* normalises the input data to zero‑mean/unit‑variance before convolution,
* learns the threshold as a trainable bias parameter,
* and exposes a `forward` method that can be used in a `torch.nn.Sequential` pipeline.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvFilter(nn.Module):
    """Drop‑in replacement for the original Conv() helper.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel.
    input_channels : int, default 1
        Number of input channels.
    output_channels : int, default 1
        Number of output channels.
    threshold : float, default 0.0
        Initial value of the learnable threshold.
    normalize : bool, default True
        Whether to normalise the input per channel.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        input_channels: int = 1,
        output_channels: int = 1,
        threshold: float = 0.0,
        *,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.normalize = normalize

        # Convolution layer with a single filter per output channel
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,  # preserve spatial size
        )

        # Learnable threshold bias per output channel
        self.threshold = nn.Parameter(
            torch.full((output_channels,), threshold, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution, thresholding, and optional normalisation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, channels, H, W)``.

        Returns
        -------
        torch.Tensor
            Output tensor after convolution and sigmoid thresholding.
        """
        if self.normalize:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            std = x.std(dim=(0, 2, 3), keepdim=True) + 1e-6
            x = (x - mean) / std

        logits = self.conv(x)
        logits = logits - self.threshold.view(1, -1, 1, 1)
        out = torch.sigmoid(logits)
        return out


def Conv() -> ConvFilter:
    """Convenience factory returning a default ConvFilter instance."""
    return ConvFilter(kernel_size=2, input_channels=1, output_channels=1)
