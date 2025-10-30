"""Hybrid convolutional filter with classical RBF kernel support.

The :class:`Gen152Conv` module keeps the original Conv filter API while
adding a kernel matrix computation that can be used in kernel‑based
learning algorithms.  It is fully classical and relies only on
PyTorch and NumPy.

Usage
-----
>>> from Conv__gen152 import Conv
>>> conv = Conv(kernel_size=3, threshold=0.5, gamma=2.0)
>>> x = torch.randn(5, 1, 28, 28)
>>> out = conv(x)          # convolution + sigmoid activation
>>> conv.set_reference(torch.randn(10, 1, 28, 28))
>>> K = conv.kernel_matrix(out)  # RBF kernel between activations and reference
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np
from typing import Optional


class Gen152Conv(nn.Module):
    """Drop‑in replacement for the original Conv filter with an RBF kernel.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolutional kernel.
    threshold : float, default 0.0
        Threshold applied after the convolution before the sigmoid.
    gamma : float, default 1.0
        RBF kernel bandwidth.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, gamma: float = 1.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.gamma = gamma
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.register_buffer("reference", torch.empty((0, 1, kernel_size, kernel_size)))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the convolution and sigmoid activation."""
        if data.ndim == 3:
            data = data.unsqueeze(0)
        conv_out = self.conv(data)
        activations = torch.sigmoid(conv_out - self.threshold)
        return activations

    def set_reference(self, ref: torch.Tensor) -> None:
        """Store a reference set for kernel evaluation."""
        self.reference = ref

    def kernel_matrix(self, x: torch.Tensor) -> np.ndarray:
        """Compute the RBF kernel matrix between ``x`` and the stored reference.

        Parameters
        ----------
        x : torch.Tensor
            Convolutional activations of shape ``(N, 1, H, W)``.

        Returns
        -------
        np.ndarray
            Kernel matrix of shape ``(N, M)`` where ``M`` is the number of
            reference samples.
        """
        if self.reference.size(0) == 0:
            raise ValueError("Reference set not set.")
        x_flat = x.view(x.size(0), -1)
        ref_flat = self.reference.view(self.reference.size(0), -1)
        diff = x_flat.unsqueeze(1) - ref_flat.unsqueeze(0)
        dist2 = torch.sum(diff * diff, dim=2)
        kernel = torch.exp(-self.gamma * dist2)
        return kernel.detach().cpu().numpy()


def Conv(kernel_size: int = 2, threshold: float = 0.0, gamma: float = 1.0) -> Gen152Conv:
    """Return a :class:`Gen152Conv` instance.

    This wrapper preserves the original ``Conv`` function name so that
    existing code can import ``Conv`` from :mod:`Conv__gen152` without
    modification.
    """
    return Gen152Conv(kernel_size=kernel_size, threshold=threshold, gamma=gamma)


__all__ = ["Conv", "Gen152Conv"]
