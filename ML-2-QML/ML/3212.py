"""Hybrid QCNN model combining classical convolution filter with fully connected network."""

from __future__ import annotations

import torch
from torch import nn


class HybridQCNN(nn.Module):
    """Classical convolution‑inspired network with a thresholded 2‑D filter."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Fully‑connected layers mirroring the original QCNN architecture
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # If 4‑D input, apply the 2‑D filter first
        if x.dim() == 4:
            conv_out = self.conv_filter(x)
            conv_out = torch.sigmoid(conv_out - self.threshold)
            conv_out = conv_out.view(conv_out.size(0), -1)  # (batch, 1)
            # Broadcast to 8 features for compatibility with the FC network
            x = conv_out.repeat(1, 8)
        # Flatten for the fully‑connected part
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


def QCNN() -> HybridQCNN:
    """Return a configured :class:`HybridQCNN` instance."""
    return HybridQCNN()


__all__ = ["QCNN", "HybridQCNN"]
