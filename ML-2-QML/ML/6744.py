"""Classical RBF kernel with optional hybrid weighting.

This module preserves the original API (`Kernel`, `KernalAnsatz`, `kernel_matrix`)
while adding a lightweight hybrid interface that can be combined with a quantum
kernel.  The hybrid kernel is parameterised by a weight `alpha` that controls
the contribution from the classical RBF component.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

__all__ = ["KernalAnsatz", "Kernel", "HybridKernel", "kernel_matrix"]


class KernalAnsatz(nn.Module):
    """Pure RBF ansatz that mimics the original compatibility layer."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Wraps :class:`KernalAnsatz` to match the original forward signature."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


class HybridKernel(nn.Module):
    """
    Hybrid kernel that linearly combines a classical RBF kernel with a quantum
    kernel supplied by a second callable.  The quantum part is expected to
    accept tensors of shape (n_samples, n_features) and return a similarity
    matrix of shape (n_samples, n_samples).
    """

    def __init__(self, gamma: float = 1.0, alpha: float = 0.5, quantum_kernel: nn.Module | None = None) -> None:
        """
        Parameters
        ----------
        gamma : float
            RBF width parameter.
        alpha : float
            Weight of the classical component (0 <= alpha <= 1).
        quantum_kernel : nn.Module, optional
            Callable that implements a quantum kernel.  If ``None`` the
            classical kernel is used exclusively.
        """
        super().__init__()
        self.classical = Kernel(gamma)
        self.alpha = alpha
        self.quantum_kernel = quantum_kernel

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Classical part
        Kc = self.classical(x, y)
        if self.quantum_kernel is None:
            return Kc
        # Quantum part
        Kq = self.quantum_kernel(x, y)
        return self.alpha * Kc + (1.0 - self.alpha) * Kq


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """
    Compute the Gram matrix between two collections of tensors using the
    original classical RBF kernel.  The function is kept for backwards
    compatibility with the seed project.
    """
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])
