"""Enhanced classical RBF kernel utilities with batched support."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class KernalAnsatz(nn.Module):
    """
    Radial basis function kernel ansatz.

    Parameters
    ----------
    gamma : float, default=1.0
        Kernel width parameter.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel for a single pair of vectors.

        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors representing data points.

        Returns
        -------
        torch.Tensor
            Scalar kernel value.
        """
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff))

    def forward_batch(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Vectorised kernel evaluation for two batches.

        Parameters
        ----------
        X : torch.Tensor of shape (n, d)
            First batch of points.
        Y : torch.Tensor of shape (m, d)
            Second batch of points.

        Returns
        -------
        torch.Tensor of shape (n, m)
            Pairwise kernel matrix.
        """
        diff = X.unsqueeze(1) - Y.unsqueeze(0)       # (n, m, d)
        dist2 = torch.sum(diff * diff, dim=-1)        # (n, m)
        return torch.exp(-self.gamma * dist2)


class Kernel(nn.Module):
    """
    Wrapper that exposes a convenient interface for kernel computation.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single pair, maintaining backward compatibility.
        """
        return self.ansatz(x, y).squeeze()

    def forward_batch(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for two batches of data.
        """
        return self.ansatz.forward_batch(X, Y)


def kernel_matrix(
    a: Sequence[torch.Tensor | np.ndarray],
    b: Sequence[torch.Tensor | np.ndarray],
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Compute the Gram matrix between two collections of points.

    Parameters
    ----------
    a, b : sequence of tensors or numpy arrays
        Data points to evaluate.
    gamma : float, default=1.0
        Kernel width.

    Returns
    -------
    np.ndarray
        Pairwise kernel matrix of shape (len(a), len(b)).
    """
    kernel = Kernel(gamma)

    # Convert inputs to torch tensors
    X = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in a])
    Y = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in b])

    return kernel.forward_batch(X, Y).detach().cpu().numpy()


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
