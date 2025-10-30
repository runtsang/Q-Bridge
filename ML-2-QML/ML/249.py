"""Enhanced classical quanvolution model.

This module defines a drop‑in replacement for the original
``QuanvolutionFilter``/``QuanvolutionClassifier`` pair.  The class
``QuanvolutionNet`` implements a single 2×2 convolution followed by
a linear classifier, mirroring the behaviour of the seed while
providing a clean, single‑class API suitable for hybrid experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionNet(nn.Module):
    """
    Classical 2×2 convolutional filter with a linear head.
    The architecture matches the original seed but exposes a
    unified interface: ``forward(x)`` returns log‑softmax logits.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 num_classes: int = 10, kernel_size: int = 2,
                 stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride)
        # After conv the feature map size is 14×14 for MNIST.
        self.linear = nn.Linear(out_channels * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, 1, 28, 28)``.
        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape ``(batch, num_classes)``.
        """
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionNet"]
