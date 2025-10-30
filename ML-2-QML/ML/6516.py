"""Hybrid kernel module with learnable RBF width and stochastic Gram matrix."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class KernalAnsatz(nn.Module):
    """Learnable RBF kernel.

    Parameters
    ----------
    gamma : float, optional
        Initial width of the RBF kernel. Stored as a learnable parameter.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel value for two batches of vectors.

        Parameters
        ----------
        x, y : torch.Tensor
            Batches of shape (..., d). The kernel is computed element‑wise
            between corresponding rows.
        """
        diff = x - y
        dist2 = torch.sum(diff * diff, dim=-1, keepdim=True)
        return torch.exp(-self.gamma * dist2)


class Kernel(nn.Module):
    """Wrapper that exposes a fit/transform interface."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return a scalar kernel value for two 1‑D tensors."""
        return self.ansatz(x, y).squeeze()

    def fit(self, X: torch.Tensor, max_iter: int = 200, lr: float = 0.01) -> "Kernel":
        """Learn the width parameter by maximizing self‑similarity."""
        optimizer = torch.optim.Adam([self.ansatz.gamma], lr=lr)
        for _ in range(max_iter):
            optimizer.zero_grad()
            K = self.forward(X, X)
            loss = -torch.mean(torch.diag(K))
            loss.backward()
            optimizer.step()
        return self

    def transform(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute the full kernel matrix between two batches."""
        K = torch.zeros((X.size(0), Y.size(0)), device=X.device, dtype=X.dtype)
        for i, x in enumerate(X):
            K[i] = self.ansatz(x.unsqueeze(0), Y).squeeze(0)
        return K


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  gamma: float = 1.0, subset_size: int | None = None) -> np.ndarray:
    """
    Compute the Gram matrix between two collections of vectors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of 1‑D tensors.
    gamma : float, optional
        Initial width of the RBF kernel.
    subset_size : int, optional
        If provided, only a random subset of ``a`` is used, yielding a
        smaller matrix useful for large‑scale experiments.
    """
    kernel = Kernel(gamma)
    if subset_size is not None:
        idx = np.random.choice(len(a), subset_size, replace=False)
        a = [a[i] for i in idx]
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
