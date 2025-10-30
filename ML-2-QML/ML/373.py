"""Enhanced classical quanvolution module with a deeper MLP head.

This module builds on the original 2×2 convolution filter but introduces
a learnable bias matrix per patch and a multi‑layer perceptron with
dropout for richer feature extraction.  The architecture is fully
trainable with PyTorch and can be dropped into any image classification
pipeline.

The class is called :class:`Quanvolution__gen413` to match the target
file name and to keep a clear mapping with the corresponding quantum
implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Quanvolution__gen413(nn.Module):
    """Hybrid classical quanvolution + MLP classifier.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 1 for MNIST).
    num_classes : int
        Number of output classes (default 10 for MNIST).
    patch_size : int
        Size of the square patch to convolve over (default 2).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        patch_size: int = 2,
    ) -> None:
        super().__init__()
        # Convolution that operates on non‑overlapping patches
        self.conv = nn.Conv2d(
            in_channels,
            out_channels=4,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,  # learnable bias per patch
        )
        # MLP head with two hidden layers and dropout
        self.mlp = nn.Sequential(
            nn.Linear(4 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, height, width).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, num_classes).
        """
        # Convolution over patches
        features = self.conv(x)  # shape (batch, 4, 14, 14)
        # Flatten for the MLP
        features = features.view(features.size(0), -1)
        logits = self.mlp(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["Quanvolution__gen413"]
