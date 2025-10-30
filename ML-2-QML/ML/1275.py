"""Enhanced classical convolutional filter with dropout and thresholding.

This module defines Conv, a drop-in replacement for the original Conv filter.
It supports optional dropout, thresholding, and multiple kernel sizes.
The run method accepts a 2D numpy array and returns a scalar activation.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Conv(nn.Module):
    """
    Classical convolutional filter with optional dropout and thresholding.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        dropout: float | None = None,
        bias: bool = True,
        padding: str = "valid",
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=bias,
            padding=padding,
            stride=stride,
        )
        if dropout is not None:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None

    def forward(self, data: np.ndarray) -> float:
        """
        Forward pass: accepts a 2D array and returns a scalar activation.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        # Ensure shape is (1, 1, H, W)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim!= 4:
            raise ValueError(f"Input tensor must be 2D or 4D, got shape {tensor.shape}")
        if self.dropout_layer is not None:
            tensor = self.dropout_layer(tensor)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

    def run(self, data: np.ndarray) -> float:
        """
        Alias for forward to match original API.
        """
        return self.forward(data)

__all__ = ["Conv"]
