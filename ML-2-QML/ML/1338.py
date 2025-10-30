"""Hybrid classical convolution filter for image patches.

The ConvFilter class implements a two‑stage pipeline:
1. A learnable 2‑D convolution (torch.nn.Conv2d) that produces a feature map.
2. A post‑processing step that converts the feature map into a probability
   value via a sigmoid and mean pooling.  The output is a scalar in [0,1].

The public API mirrors the original Conv() factory but now returns a
``ConvFilter`` instance that is fully trainable with PyTorch optimizers.
"""

from __future__ import annotations

import torch
from torch import nn

class ConvFilter(nn.Module):
    """Hybrid convolution‑quantum filter (classical variant)."""

    def __init__(
        self,
        kernel_size: int = 2,
        conv_bias: bool = True,
        conv_stride: int = 1,
        conv_padding: int = 0,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(
            1,
            1,
            kernel_size=kernel_size,
            bias=conv_bias,
            stride=conv_stride,
            padding=conv_padding,
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid filter.

        Parameters
        ----------
        data : torch.Tensor
            Input image patch of shape (H, W) or (1, H, W).

        Returns
        -------
        torch.Tensor
            Scalar probability in [0, 1].
        """
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

def Conv():
    """Return a callable object that emulates the quantum filter with PyTorch ops."""
    return ConvFilter()

__all__ = ["Conv", "ConvFilter"]
