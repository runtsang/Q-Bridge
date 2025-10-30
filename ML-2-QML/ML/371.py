"""Hybrid classical RBF kernel with trainable parameters.

This module defines :class:`QuantumKernelMethod` that implements a
trainable radial‑basis‑function kernel.  It is deliberately
compatible with the quantum interface used in the QML seed so that
both implementations can be swapped in downstream experiments.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence

__all__ = ["QuantumKernelMethod", "kernel_matrix"]

class QuantumKernelMethod(nn.Module):
    """Classical RBF kernel with learnable width.

    Parameters
    ----------
    gamma : float | None
        Initial kernel width.  If ``None`` a default of ``sqrt(2)`` is
        used.  The value is stored as a ``nn.Parameter`` so that it can
        be optimized during training.
    bias : bool, optional
        If ``True`` a bias term is added to the kernel value.
    """
    def __init__(self, gamma: float | None = None, *, bias: bool = False) -> None:
        super().__init__()
        init_gamma = np.sqrt(2.0) if gamma is None else gamma
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the RBF kernel value for two 1‑D tensors.

        The tensors are expected to be flat vectors.  The method
        returns a scalar ``torch.Tensor`` containing the kernel value.
        """
        diff = x - y
        sq_norm = (diff ** 2).sum()
        k = torch.exp(-self.gamma * sq_norm)
        if self.bias:
            k = k + self.bias_param
        return k

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float | None = None) -> np.ndarray:
    """Compute the Gram matrix for two sequences of tensors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of 1‑D tensors.  The kernel is evaluated pairwise
        between elements of ``a`` and ``b``.
    gamma : float | None, optional
        Initial gamma value for the :class:`QuantumKernelMethod`
        instance.  ``None`` defaults to ``sqrt(2)``.

    Returns
    -------
    np.ndarray
        2‑D array of shape ``(len(a), len(b))`` containing the kernel
        values.
    """
    model = QuantumKernelMethod(gamma=gamma)
    return np.array([[model(x, y).item() for y in b] for x in a])
