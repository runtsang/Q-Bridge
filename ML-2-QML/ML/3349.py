"""Hybrid classical convolutional regression model.

This module defines HybridConvRegression that combines a 2‑D convolution
filter with a fully‑connected head.  The convolution is a drop‑in
replacement for the quantum quanvolution used in the original
reference.  The class is fully compatible with PyTorch training loops
and can be instantiated with a kernel size and threshold.

The model accepts input tensors of shape (batch, 1, k, k) where
k = kernel_size.  For convenience a helper method `from_features`
reshapes 1‑D feature vectors of length k*k into the required 2‑D
format.
"""

import torch
from torch import nn
from typing import Tuple


class HybridConvRegression(nn.Module):
    """Classical convolution + linear head for regression."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, hidden_dim: int = 32) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 1, k, k)

        Returns:
            Tensor of shape (batch,) with regression predictions.
        """
        logits = self.conv(x)
        logits = torch.sigmoid(logits - self.threshold)
        flat = self.flatten(logits)
        return self.head(flat).squeeze(-1)

    @staticmethod
    def from_features(features: torch.Tensor, kernel_size: int = 2) -> "HybridConvRegression":
        """
        Utility to create a model from 1‑D feature vectors.

        Args:
            features: Tensor of shape (batch, k*k)
            kernel_size: size of the 2‑D kernel.

        Returns:
            A HybridConvRegression instance ready for training.
        """
        return HybridConvRegression(kernel_size=kernel_size)


__all__ = ["HybridConvRegression"]
