"""Hybrid QCNNGen182: classical and quantum inspired network.

This module defines a hybrid QCNNGen182 class that combines a classical
convolutional filter (ConvFilter) inspired by the Conv.py seed with a
fully‑connected QCNN architecture derived from QCNNModel.  The network
supports optional pooling, configurable layer sizes, and a thresholded
activation that mimics the quantum filter behaviour.

The class can be instantiated with a dictionary of hyper‑parameters
allowing easy experimentation and comparison against the original
QCNNModel.
"""

from __future__ import annotations

import math
import torch
from torch import nn
from typing import List, Optional

# --------------------------------------------------------------------------- #
#  Classical convolutional filter (inspired by Conv.py ML seed)
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """Simple 2‑D convolution filter with a thresholded sigmoid activation.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square kernel.
    threshold : float, default 0.0
        Threshold applied before the sigmoid.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the filter.

        Parameters
        ----------
        data : torch.Tensor
            Input tensor of shape (batch, height, width) or (height, width).

        Returns
        -------
        torch.Tensor
            Filtered output of shape (batch, 1, 1, 1) after sigmoid.
        """
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        else:
            data = data.unsqueeze(1)  # (B,1,H,W)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations

# --------------------------------------------------------------------------- #
#  Hybrid QCNN architecture
# --------------------------------------------------------------------------- #
class QCNNGen182(nn.Module):
    """Hybrid QCNNGen182 model combining ConvFilter and fully‑connected layers.

    The architecture is a stack of linear layers with optional pooling
    operations.  The first layer can be replaced by a ConvFilter to
    emulate a quantum convolutional filter.

    Parameters
    ----------
    input_dim : int
        Size of the input feature vector.
    conv_kernel_size : int, optional
        Kernel size for the ConvFilter.  If ``None`` the filter is omitted.
    conv_threshold : float, optional
        Threshold for the ConvFilter.
    layer_sizes : List[int], optional
        Sizes of the hidden layers.  Defaults to [16, 16, 8, 4, 4].
    pool_sizes : List[int], optional
        Sizes of the pooling layers.  If ``None`` pooling is omitted.
    """

    def __init__(
        self,
        input_dim: int,
        conv_kernel_size: Optional[int] = 2,
        conv_threshold: float = 0.0,
        layer_sizes: Optional[List[int]] = None,
        pool_sizes: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.conv_filter: Optional[ConvFilter] = None
        if conv_kernel_size is not None:
            # The filter expects a 2‑D input; we reshape the flat vector
            # into a square matrix if possible.
            side = int(math.sqrt(input_dim))
            if side * side!= input_dim:
                raise ValueError(
                    "input_dim must be a perfect square when using ConvFilter"
                )
            self.conv_filter = ConvFilter(kernel_size=conv_kernel_size, threshold=conv_threshold)

        # Build the fully‑connected body
        sizes = layer_sizes or [16, 16, 8, 4, 4]
        layers: List[nn.Module] = []
        in_features = input_dim if self.conv_filter is None else 1
        for out_features in sizes:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.Tanh())
            in_features = out_features

        # Optional pooling
        if pool_sizes:
            pool_layers: List[nn.Module] = []
            for size in pool_sizes:
                pool_layers.append(nn.Linear(in_features, size))
                pool_layers.append(nn.Tanh())
                in_features = size
            layers.extend(pool_layers)

        layers.append(nn.Linear(in_features, 1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch, 1).
        """
        if self.conv_filter is not None:
            # Reshape to 2‑D, apply filter, then flatten
            batch = x.shape[0]
            side = int(math.sqrt(x.shape[1]))
            x = x.view(batch, 1, side, side)
            x = self.conv_filter(x)
            x = x.view(batch, -1)  # flatten
        return torch.sigmoid(self.body(x))

# --------------------------------------------------------------------------- #
#  Factory function
# --------------------------------------------------------------------------- #
def QCNNGen182Factory(**kwargs) -> QCNNGen182:
    """Return a configured QCNNGen182 instance.

    The factory accepts the same keyword arguments as ``QCNNGen182``.
    """
    return QCNNGen182(**kwargs)

__all__ = ["QCNNGen182", "QCNNGen182Factory", "ConvFilter"]
