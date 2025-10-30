"""Hybrid kernel method combining convolutional preprocessing and RBF kernel.

This module implements a drop‑in replacement for the original
`QuantumKernelMethod.py`.  It exposes the same public API but
adds a classical convolutional feature extractor before the RBF
kernel, mirroring the quantum quanvolution in the QML counterpart.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

__all__ = ["Conv", "KernalAnsatz", "HybridKernel", "kernel_matrix"]


# --------------------------------------------------------------------------- #
#  Classical convolutional filter (drop‑in for quanvolution)
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """2‑D convolutional filter that emulates the quantum filter.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square filter.
    threshold : float, default 0.0
        Threshold applied before the sigmoid activation.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the filter to 2‑D data and return a scalar feature.

        Parameters
        ----------
        data
            Tensor of shape ``(kernel_size, kernel_size)``.
        """
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size).float()
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()


def Conv(kernel_size: int = 2, threshold: float = 0.0) -> ConvFilter:
    """Factory that returns an instance of :class:`ConvFilter`."""
    return ConvFilter(kernel_size=kernel_size, threshold=threshold)


# --------------------------------------------------------------------------- #
#  RBF kernel implementation
# --------------------------------------------------------------------------- #
class KernalAnsatz(nn.Module):
    """Radial basis function (RBF) kernel.

    Parameters
    ----------
    gamma : float, default 1.0
        Width parameter of the RBF.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


# --------------------------------------------------------------------------- #
#  Hybrid kernel that chains convolution with the RBF
# --------------------------------------------------------------------------- #
class HybridKernel(nn.Module):
    """Hybrid kernel that first extracts local features via a convolution
    filter and then applies an RBF kernel on the resulting scalars.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        conv_kernel_size: int = 2,
        conv_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = Conv(kernel_size=conv_kernel_size, threshold=conv_threshold)
        self.kernel = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the hybrid kernel between two batches.

        Parameters
        ----------
        x, y
            Tensors of shape ``(batch, features)`` where ``features`` equals
            ``conv_kernel_size ** 2``.  Each row is treated as a 2‑D image.
        """
        # Convert each row into a 2‑D image and apply the filter
        x_feat = torch.stack(
            [
                self.conv.run(
                    xi.reshape(-1, self.conv.kernel_size, self.conv.kernel_size)
                )
                for xi in x
            ]
        )
        y_feat = torch.stack(
            [
                self.conv.run(
                    yi.reshape(-1, self.conv.kernel_size, self.conv.kernel_size)
                )
                for yi in y
            ]
        )
        return self.kernel(x_feat, y_feat)


# --------------------------------------------------------------------------- #
#  Utility to compute a Gram matrix
# --------------------------------------------------------------------------- #
def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two sequences of tensors.

    Parameters
    ----------
    a, b
        Sequences of tensors; each tensor is expected to be 1‑D.
    gamma
        RBF width parameter.
    """
    kernel = HybridKernel(gamma=gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])
