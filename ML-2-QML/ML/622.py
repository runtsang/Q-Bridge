"""Enhanced classical convolutional filter with batch support and autograd."""

from __future__ import annotations

import torch
from torch import nn
import numpy as np


def Conv() -> "ConvFilter":
    """Return a callable object that emulates the quantum filter with PyTorch ops."""
    return ConvFilter()


class ConvFilter(nn.Module):
    """
    A drop‑in replacement for the original Conv filter.

    Features
    --------
    - Supports batched input tensors of shape (B, C, H, W).
    - Provides a full nn.Module interface for autograd and integration into larger models.
    - Thresholding and sigmoid activation are applied element‑wise after the convolution.
    - Exposes a convenient ``run`` method that accepts NumPy arrays.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square convolution kernel.
    threshold : float, default 0.0
        Bias value subtracted before the sigmoid activation.
    bias : bool, default True
        Whether the underlying convolution has a learnable bias term.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=bias,
        )
        # Initialise weights for reproducibility
        nn.init.kaiming_uniform_(self.conv.weight, a=np.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.conv.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W) or (1, 1, H, W).

        Returns
        -------
        torch.Tensor
            Mean activation across the batch and spatial dimensions.
        """
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=(0, 2, 3))

    def run(self, data: np.ndarray) -> float:
        """
        Convenience wrapper that accepts a NumPy array.

        Parameters
        ----------
        data : np.ndarray
            2‑D array with shape (kernel_size, kernel_size) or a larger image.

        Returns
        -------
        float
            Mean activation value.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return self.forward(tensor).item()


__all__ = ["Conv", "ConvFilter"]
