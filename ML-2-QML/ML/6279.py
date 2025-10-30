"""Hybrid QCNN with classical convolution filter and deep quantum-inspired layers.

This module defines a PyTorch model that integrates a lightweight classical convolution
filter (emulating the quantum quanvolution) with a stack of fully‑connected layers
mirroring the quantum convolution‑pooling sequence from the original QCNN.
"""

from __future__ import annotations

import torch
from torch import nn

class ConvFilter(nn.Module):
    """A simple 2×2 convolution filter with a sigmoid activation.

    The filter is deliberately small to keep the classical pre‑processing cheap
    while still providing a non‑linear transformation of the raw data.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, H, W)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations

class QCNNGen088Model(nn.Module):
    """Classical head + quantum‑inspired convolutional stack.

    The model first applies a 2×2 ConvFilter, then flattens the result
    and passes it through a sequence of linear layers that mimic the
    quantum convolution, pooling and final readout from the original QCNN.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv_filter = ConvFilter(kernel_size=2, threshold=0.0)

        # The feature map size after the 2×2 convolution on an 8×8 input
        # is 7×7 = 49, but we keep the architecture flexible.
        self.feature_map = nn.Sequential(
            nn.Linear(49, 16),
            nn.Tanh()
        )
        self.conv1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.Tanh()
        )
        self.pool1 = nn.Sequential(
            nn.Linear(16, 12),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Linear(12, 8),
            nn.Tanh()
        )
        self.pool2 = nn.Sequential(
            nn.Linear(8, 4),
            nn.Tanh()
        )
        self.conv3 = nn.Sequential(
            nn.Linear(4, 4),
            nn.Tanh()
        )
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 8, 8)
        x = self.conv_filter(x)          # (batch, 1, 7, 7)
        x = x.view(x.size(0), -1)        # flatten to (batch, 49)
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

def QCNNGen088() -> QCNNGen088Model:
    """Factory returning a pre‑configured :class:`QCNNGen088Model`."""
    return QCNNGen088Model()

__all__ = ["QCNNGen088", "QCNNGen088Model"]
