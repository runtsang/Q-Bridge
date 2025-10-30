"""
Hybrid classical convolutional filter with residuals, learnable kernels,
and a drop‑in `run` interface that matches the original Conv() callable.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from typing import Iterable

__all__ = ["ConvEnhanced"]

class ConvEnhanced(nn.Module):
    """
    Multi‑channel, learnable 2‑D convolutional filter with optional residual
    connection and sigmoid activation.  The module can be used as a drop‑in
    replacement for the original ``Conv()`` callable.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolutional kernel.
    threshold : float, default 0.0
        Bias applied before the sigmoid activation.
    in_channels : int, default 1
        Number of input channels.
    out_channels : int, default 1
        Number of output channels.
    bias : bool, default True
        Whether to add a bias term to the convolution.
    residual : bool, default True
        Whether to add a skip connection from the input to the output.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        in_channels: int = 1,
        out_channels: int = 1,
        bias: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual

        # Learnable convolutional layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
        )

    def run(self, data: Iterable[float]) -> float:
        """
        Apply the convolutional filter to a single kernel-sized patch.

        Parameters
        ----------
        data : array‑like
            2‑D array with shape (kernel_size, kernel_size) or a flattened
            vector of length kernel_size**2.

        Returns
        -------
        float
            The mean sigmoid activation over the output channels.
        """
        # Convert input to tensor and reshape
        arr = torch.as_tensor(data, dtype=torch.float32)
        if arr.ndim == 1:
            arr = arr.reshape(self.kernel_size, self.kernel_size)
        arr = arr.unsqueeze(0).unsqueeze(0)  # shape (1, 1, k, k)

        # Forward pass
        logits = self.conv(arr)  # shape (1, out_channels, k, k)
        if self.residual:
            # Pad input to match output dimensions if necessary
            if self.in_channels == self.out_channels:
                logits = logits + arr
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward method for compatibility with nn.Module.
        """
        logits = self.conv(x)
        if self.residual:
            logits = logits + x
        return torch.sigmoid(logits - self.threshold)
