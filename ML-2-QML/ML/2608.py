"""Hybrid Quanvolution–QCNN classifier.

This module combines a classical 2×2 convolutional filter with a QCNN-inspired
fully‑connected backbone.  The convolution reduces the 28×28 MNIST image to a
4‑channel feature map (size 14×14), which is flattened and then processed
through a sequence of linear layers that mimic the convolution, pooling and
down‑sampling stages of a QCNN.  The final linear head outputs class logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionQCNNClassifier(nn.Module):
    """Hybrid classifier that merges classical quanvolution and QCNN ideas."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        # Classical 2×2 convolution with stride 2 – 4 output channels
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=2, stride=2)

        # QCNN‑style fully connected backbone
        self.feature_map = nn.Sequential(nn.Linear(4 * 14 * 14, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x
            Input image tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax of class logits.
        """
        # Classical convolution
        x = self.conv(x)
        x = x.view(x.size(0), -1)

        # QCNN‑style processing
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        logits = self.head(x)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionQCNNClassifier"]
