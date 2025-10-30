"""
ConvEnhanced: Classical hybrid convolutional filter.

This module implements ConvEnhanced as a torch.nn.Module that
consists of:
- A 2D convolution with a single input and output channel.
- A small MLP head that maps the convolution output to a scalar.
- An optional learnable threshold applied before the sigmoid.

The class exposes a run method that accepts a 2‑D array or a
torch.Tensor and returns a scalar float.  It is fully differentiable
and ready to be plugged into larger CNN pipelines.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np

class ConvEnhanced(nn.Module):
    """
    Args:
        kernel_size (int): Size of the convolution kernel.
        threshold (float): Initial threshold applied before sigmoid.
        learnable_threshold (bool): Whether to learn the threshold.
        hidden_dim (int): Size of the hidden layer in the MLP head.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 learnable_threshold: bool = True,
                 hidden_dim: int = 16) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(1, 1,
                              kernel_size=kernel_size,
                              bias=False)
        self.sigmoid = nn.Sigmoid()
        if learnable_threshold:
            self.threshold = nn.Parameter(
                torch.tensor(threshold, dtype=torch.float32))
        else:
            self.threshold = torch.tensor(threshold, dtype=torch.float32)
        # MLP head mapping a single scalar to a scalar
        self.head = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that accepts a 2‑D tensor of shape
        (kernel_size, kernel_size) or a batch of such tensors.
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        if data.dim() == 2:
            data = data.unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
        elif data.dim() == 3:
            # batch of (h,w)
            data = data.unsqueeze(1)
        # Convolution
        out = self.conv(data)  # (batch,1,1,1)
        out = out.view(-1, 1)   # (batch,1)
        out = self.sigmoid(out - self.threshold)
        out = self.head(out)    # (batch,1)
        return out.squeeze()

    def run(self, data):
        """Convenience method that returns a plain Python float."""
        with torch.no_grad():
            return float(self.forward(data).item())

__all__ = ["ConvEnhanced"]
