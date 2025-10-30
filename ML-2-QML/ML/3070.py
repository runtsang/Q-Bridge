"""Hybrid QCNN – classical implementation.

The model combines a patch‑wise 2×2 convolution (mimicking the quanvolution filter)
with two stages of 2‑D pooling and a fully connected head.  It is fully
trainable with standard PyTorch optimisers.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class HybridQCNN(nn.Module):
    """Classical hybrid QCNN.

    Architecture:
      * Patch‑wise 2×2 convolution (4 output channels) – inspired by quanvolution.
      * Two stages of 2‑D max‑pooling (stride 2) – reduces spatial resolution.
      * Fully connected head with log‑softmax output for multi‑class classification.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.patch_conv = nn.Conv2d(in_channels, 4, kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Compute flattened feature size for the linear head
        dummy = torch.zeros(1, in_channels, 28, 28)
        feat = self.pool2(self.pool1(self.patch_conv(dummy)))
        self.feature_dim = feat.numel()

        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.patch_conv(x)
        x = self.pool1(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        logits = self.head(x)
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridQCNN"]
