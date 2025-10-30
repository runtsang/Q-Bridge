"""Enhanced classical convolutional filter with trainable threshold and optional depth.

This module defines ConvEnhanced, a drop‑in replacement for the original Conv filter.
It adds depth‑wise separable convolution, optional batch‑norm, ReLU, and a learnable
threshold that is updated during back‑propagation.

The class exposes a `forward` method that accepts a numpy array or torch tensor
and returns a scalar activation, matching the interface of the original library.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ConvEnhanced(nn.Module):
    """
    A configurable convolutional filter.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolutional kernel.
    depth : int, default 1
        Number of depth‑wise separable convolution layers.
    use_batchnorm : bool, default True
        Whether to apply batch‑normalization after each layer.
    dropout : float, default 0.0
        Drop‑out probability between layers.
    device : str | None, default None
        Device to place the model on.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        depth: int = 1,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.depth = max(1, depth)
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Build a depth‑wise separable convolution stack
        layers = []
        in_channels = 1
        for _ in range(self.depth):
            conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, bias=True)
            layers.append(conv)
            if self.use_batchnorm:
                layers.append(nn.BatchNorm2d(in_channels))
            layers.append(nn.ReLU(inplace=True))
            if self.dropout > 0.0:
                layers.append(nn.Dropout2d(p=self.dropout))
        self.conv_stack = nn.Sequential(*layers)

        # Learnable threshold
        self.threshold = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.to(self.device)

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Forward pass that returns a scalar activation.

        Parameters
        ----------
        x : torch.Tensor or numpy.ndarray
            Input data of shape (H, W) or (N, H, W) with single channel.

        Returns
        -------
        torch.Tensor
            Scalar activation value.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        elif x.ndim == 3:
            x = x.unsqueeze(1)  # (N,1,H,W)
        x = x.to(self.device)

        logits = self.conv_stack(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

    def run(self, data: np.ndarray | torch.Tensor) -> float:
        """
        Convenience method to run the filter on CPU and return a Python float.
        """
        with torch.no_grad():
            out = self.forward(data)
        return out.item()

__all__ = ["ConvEnhanced"]
