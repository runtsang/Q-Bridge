"""
Enhanced classical convolutional network with optional dropout and dense head.
Provides a drop‑in replacement for the original Conv filter while adding
layer‑wise normalisation and a simple classification head.

The design keeps the original public signature Conv() so that downstream
code can be swapped in without changes.  The returned object is a nn.Module
that can be trained end‑to‑end with any optimizer and loss.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class ConvNetDual(nn.Module):
    """
    Small CNN that emulates the original 2‑D convolution filter but with
    batch‑normalisation, dropout and a dense head.  The network operates
    on single‑channel images and returns a single scalar output.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.device = device

        # Convolutional feature extractor
        self.conv = nn.Conv2d(1, 8, kernel_size=kernel_size, padding=0)
        self.bn = nn.BatchNorm2d(8)

        # Dense head
        flattened = 8 * kernel_size * kernel_size
        self.fc = nn.Linear(flattened, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W) or (H, W).  If the input
            is 2‑D, a batch and channel dimension are added automatically.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, 1) containing raw logits.
        """
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3 and x.shape[1]!= 1:
            # Assume shape (B, H, W)
            x = x.unsqueeze(1)

        # Apply thresholding and sigmoid
        x = torch.sigmoid(x - self.threshold)

        # Convolution + batch‑norm + non‑linearity
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)

        # Flatten and dense head
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.out(x)
        return x

    def run(self, data: np.ndarray | list) -> float:
        """
        Convenience wrapper that accepts raw 2‑D data and returns a scalar.

        Parameters
        ----------
        data : np.ndarray | list
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Scalar output after a forward pass.
        """
        with torch.no_grad():
            tensor = torch.tensor(data, dtype=torch.float32)
            output = self.forward(tensor)
            return output.item()

def Conv() -> ConvNetDual:
    """
    Factory function that returns a ConvNetDual instance with default
    hyper‑parameters.  It mimics the original Conv() API for backward
    compatibility.
    """
    return ConvNetDual()

__all__ = ["ConvNetDual", "Conv"]
