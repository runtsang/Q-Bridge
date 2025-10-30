"""Enhanced classical radial basis function kernel with trainable parameters."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

class QuantumKernelMethod(nn.Module):
    """Trainable RBF kernel with a learnable diagonal weight matrix.

    The kernel is defined as:
        k(x, y) = exp(-γ * ||W(x - y)||²)
    where γ is a fixed hyper‑parameter and W is a learnable diagonal weight
    matrix (or full matrix if ``full=True``).  This allows the kernel to adapt
    to the feature space while keeping the classical RBF structure.
    """
    def __init__(self, gamma: float = 1.0, full: bool = False) -> None:
        super().__init__()
        self.gamma = gamma
        self.full = full
        self.W = None  # will be initialized on first forward

    def _init_W(self, x: torch.Tensor) -> None:
        if self.W is None:
            dim = x.shape[-1]
            if self.full:
                self.W = nn.Parameter(torch.eye(dim))
            else:
                self.W = nn.Parameter(torch.ones(dim))
            self.register_parameter("W", self.W)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel value for two batches of samples.

        Parameters
        ----------
        x : torch.Tensor
            Batch of samples of shape (n, d).
        y : torch.Tensor
            Batch of samples of shape (m, d).

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (n, m).
        """
        self._init_W(x)
        # Apply the weight matrix
        if self.full:
            wx = x @ self.W
            wy = y @ self.W
        else:
            wx = x * self.W
            wy = y * self.W
        diff = wx.unsqueeze(1) - wy.unsqueeze(0)  # (n, m, d)
        sq_norm = torch.sum(diff ** 2, dim=-1)
        return torch.exp(-self.gamma * sq_norm)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0, full: bool = False) -> np.ndarray:
    """
    Compute the Gram matrix between two sequences of tensors using the
    :class:`QuantumKernelMethod` kernel.

    Parameters
    ----------
    a : Sequence[torch.Tensor]
        First sequence of tensors.
    b : Sequence[torch.Tensor]
        Second sequence of tensors.
    gamma : float, optional
        Hyper‑parameter for the RBF kernel.
    full : bool, optional
        Whether to use a full weight matrix (True) or a diagonal one (False).

    Returns
    -------
    np.ndarray
        Gram matrix of shape (len(a), len(b)).
    """
    kernel = QuantumKernelMethod(gamma=gamma, full=full)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["QuantumKernelMethod", "kernel_matrix"]
