"""Hybrid QCNN – classical implementation.

This module builds a classical analogue of the quantum QCNN using
convolution‑like fully‑connected layers and a thresholded 2×2 ConvFilter.
The architecture mirrors the quantum depth, but remains purely
PyTorch‑based.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvFilter(nn.Module):
    """2×2 convolutional filter with a learnable threshold.

    The filter emulates a quantum quanvolution layer by applying a
    single 2×2 kernel followed by a sigmoid activation shifted by a
    threshold.  The kernel is trainable and the threshold allows the
    model to learn a decision boundary on the raw input intensities.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x assumed shape (B, H, W) – add channel dim
        x = x.unsqueeze(1)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=(2, 3))  # global average over spatial dims


class QCNNHybrid(nn.Module):
    """Classical QCNN architecture.

    The network consists of:
    1. A learnable 2×2 ConvFilter (emulating a quantum filter).
    2. Three convolution‑like fully‑connected blocks with Tanh non‑linearity.
    3. Two pooling layers that reduce dimensionality (simulating quantum
       pooling).
    4. A final linear head producing a single sigmoid‑activated output.
    """
    def __init__(self,
                 conv_kernel: int = 2,
                 threshold: float = 0.0,
                 hidden_dims: tuple[int, int, int] = (16, 12, 8),
                 pool_dims: tuple[int, int] = (4, 2)) -> None:
        super().__init__()
        self.filter = ConvFilter(kernel_size=conv_kernel, threshold=threshold)

        # Fully‑connected “convolution” blocks
        self.fc1 = nn.Sequential(nn.Linear(1, hidden_dims[0]), nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[1]), nn.Tanh())
        self.fc3 = nn.Sequential(nn.Linear(hidden_dims[1], hidden_dims[2]), nn.Tanh())

        # Pooling (dimensionality reduction) layers
        self.pool1 = nn.Sequential(nn.Linear(hidden_dims[2], pool_dims[0]), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(pool_dims[0], pool_dims[1]), nn.Tanh())

        self.head = nn.Linear(pool_dims[1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, H, W)
        x = self.filter(x)          # (B, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.head(x)
        return torch.sigmoid(x)


def QCNN() -> QCNNHybrid:
    """Factory returning a classical QCNNHybrid instance."""
    return QCNNHybrid()


__all__ = ["QCNNHybrid", "QCNN"]
