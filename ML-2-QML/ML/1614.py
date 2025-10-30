"""Enhanced classical RBF kernel with automatic bandwidth and batching."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]


class KernalAnsatz(nn.Module):
    """Classical RBF kernel ansatz with optional automatic gamma estimation."""

    def __init__(self, gamma: float | None = None) -> None:
        super().__init__()
        self.gamma = gamma

    def _estimate_gamma(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Median heuristic on pairwise distances."""
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # shape [m, n, d]
        dist_sq = torch.sum(diff ** 2, dim=-1)
        median = torch.median(dist_sq)
        return 1.0 / (2.0 * median.item() + 1e-6)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return RBF kernel value between two 1‑D vectors."""
        if self.gamma is None:
            self.gamma = self._estimate_gamma(x[None, :], y[None, :])
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff))


class Kernel(nn.Module):
    """Wrapper that exposes a forward interface compatible with the quantum Kernel."""

    def __init__(self, gamma: float | None = None) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)


def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    gamma: float | None = None,
) -> np.ndarray:
    """Compute Gram matrix between two collections of 1‑D tensors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Collections of vectors of equal dimension.
    gamma : float, optional
        Bandwidth parameter; if None, it is estimated from ``a``.
    """
    kernel = Kernel(gamma)
    a = [x.view(-1) for x in a]
    b = [y.view(-1) for y in b]
    return np.array([[kernel(x, y).item() for y in b] for x in a])
