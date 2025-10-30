"""Hybrid Quanvolution model with classical convolution and fully‑connected head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionHybrid(nn.Module):
    """
    Classical implementation of a hybrid quanvolution network.

    The network first extracts 2×2 patches via a convolutional layer with stride 2,
    then passes the flattened feature map through a deep fully‑connected head.
    Drop‑out and batch‑normalisation are used to improve generalisation.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        # 2×2 patch extraction
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=2, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.dropout = nn.Dropout2d(p=0.2)

        # Fully‑connected head
        self.fc = nn.Sequential(
            nn.Linear(4 * 14 * 14, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
