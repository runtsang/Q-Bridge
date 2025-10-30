"""Hybrid classical‑quantum kernel module with advanced utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

__all__ = ["HybridKernel", "kernel_matrix"]


class HybridKernel(nn.Module):
    """
    A hybrid kernel that blends a classical RBF kernel with a quantum kernel.

    Parameters
    ----------
    gamma : float, default 1.0
        Width parameter for the RBF kernel.
    alpha : float, default 0.5
        Weight of the quantum kernel in the final kernel value.
    use_quantum : bool, default False
        If ``True`` the kernel will attempt to call a quantum kernel function.
        In the pure classical implementation this flag is ignored and the
        quantum contribution is set to zero.
    """

    def __init__(self, gamma: float = 1.0, alpha: float = 0.5, use_quantum: bool = False) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.use_quantum = use_quantum

    def rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the classical RBF kernel."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the hybrid kernel.

        For the classical part we compute the RBF kernel.  If
        ``use_quantum`` is ``True`` the user can inject a callable
        ``quantum_kernel`` that receives the same inputs and returns
        a scalar tensor.  The final value is a weighted sum
        ``(1‑α) * rbf + α * quantum``.
        """
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        # Classical RBF part
        rbf_val = self.rbf(x, y)

        if self.use_quantum:
            # The quantum kernel is expected to be passed via a global
            # function or via subclassing.  Here we simply raise an error
            # to keep the implementation purely classical.
            raise NotImplementedError(
                "Quantum kernel integration is not available in the classical module."
            )

        # Return the classical kernel for the default case
        return rbf_val.squeeze()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """
    Compute the Gram matrix for a list of samples ``a`` against ``b`` using the
    hybrid kernel.  The quantum part is omitted in the classical module.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of 1‑D tensors of equal dimensionality.
    gamma : float, default 1.0
        Width parameter for the RBF kernel.

    Returns
    -------
    np.ndarray
        The Gram matrix of shape (len(a), len(b)).
    """
    kernel = HybridKernel(gamma=gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])
