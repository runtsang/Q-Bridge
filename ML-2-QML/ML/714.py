"""
ConvEnhanced: Classical convolution with learnable per‑patch mask.
Provides a drop‑in replacement for the original Conv class while adding
multi‑resolution support and a trainable mask that modulates the convolution
output.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class _ConvEnhanced(nn.Module):
    """
    Classical hybrid convolutional filter.
    Applies a 2×2 convolution and multiplies the result by a learnable
    mask derived from the input patches.
    """
    def __init__(self, kernel_size: int = 2, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=True)
        # Unfold to extract patches
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=1)
        # Mask network: simple linear layer mapping flattened patch to scalar
        self.mask_net = nn.Linear(kernel_size * kernel_size, 1)
        # Optional bias for the mask
        self.mask_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, in_channels, H, W)

        Returns:
            Tensor of shape (batch, out_channels, H-k+1, W-k+1) after mask modulation.
        """
        conv_out = self.conv(x)  # shape (batch, out_channels, H-k+1, W-k+1)

        # Extract overlapping patches
        patches = self.unfold(x)  # shape (batch, kernel_size*kernel_size, L)

        # Compute a mask per patch
        mask = torch.sigmoid(self.mask_net(patches) + self.mask_bias)  # shape (batch, 1, L)

        # Reshape mask to match conv_out
        mask = mask.view_as(conv_out)

        # Modulate convolution output by the mask
        return conv_out * mask

def ConvEnhanced():
    """
    Factory returning a ConvEnhanced instance.
    """
    return _ConvEnhanced()

__all__ = ["ConvEnhanced"]
