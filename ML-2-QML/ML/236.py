"""
Hybrid classical kernel module.

This module extends the original radial‑basis‑function kernel by:
* Combining RBF and polynomial kernels with user‑specified weights.
* Supporting GPU tensors for large‑scale data.
* Providing a convenient ``gram`` method to return the full Gram matrix.

The public API mirrors the original while offering richer functionality.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
import torch
from torch import nn


class HybridKernel(nn.Module):
    """
    Hybrid kernel combining RBF and polynomial terms.

    Parameters
    ----------
    gamma : float, default 1.0
        Width parameter for the RBF component.
    poly_coef : float, default 0.0
        Weight of the polynomial component.
    poly_degree : int, default 3
        Degree of the polynomial kernel.
    device : torch.device or str, optional
        Target device for computation (e.g. ``'cpu'`` or ``'cuda'``).

    Notes
    -----
    The kernel value is computed as

        K(x, y) = exp(-gamma * ||x - y||^2) + poly_coef * (x · y + 1)^poly_degree
    """
    def __init__(
        self,
        gamma: float = 1.0,
        *,
        poly_coef: float = 0.0,
        poly_degree: int = 3,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.poly_coef = poly_coef
        self.poly_degree = poly_degree
        self.device = torch.device(device or "cpu")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the hybrid kernel value for a pair of vectors.

        Parameters
        ----------
        x, y : torch.Tensor
            Input vectors of shape ``(N, D)`` and ``(M, D)`` respectively.

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape ``(N, M)``.
        """
        x = x.to(self.device).view(-1, x.shape[-1])
        y = y.to(self.device).view(-1, y.shape[-1])

        # RBF component
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        rbf = torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1))

        # Polynomial component
        if self.poly_coef!= 0.0:
            poly = self.poly_coef * (torch.matmul(x, y.t()) + 1) ** self.poly_degree
            return rbf + poly
        else:
            return rbf

    def gram(self, X: torch.Tensor, Y: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute the Gram matrix for datasets X and Y.

        Parameters
        ----------
        X : torch.Tensor
            Shape ``(N, D)``.
        Y : torch.Tensor, optional
            Shape ``(M, D)``.  If ``None``, Y = X.

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape ``(N, M)``.
        """
        Y = Y if Y is not None else X
        return self.forward(X, Y)


def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    *,
    gamma: float = 1.0,
    poly_coef: float = 0.0,
    poly_degree: int = 3,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """
    Compute the Gram matrix between two sequences of tensors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of 1‑D tensors.
    gamma, poly_coef, poly_degree, device : see :class:`HybridKernel`.

    Returns
    -------
    np.ndarray
        NumPy array of shape ``(len(a), len(b))``.
    """
    kernel = HybridKernel(gamma, poly_coef=poly_coef, poly_degree=poly_degree, device=device)
    return kernel.gram(torch.stack(a), torch.stack(b)).cpu().numpy()


__all__ = ["HybridKernel", "kernel_matrix"]
