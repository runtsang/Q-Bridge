"""Enhanced classical kernel module for rapid prototyping and hyper‑parameter tuning."""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np
import torch
from torch import nn


def _default_rbf(gamma: float) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def rbf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-gamma * torch.sum(diff * diff, dim=-1, keepdim=True))
    return rbf


def _polynomial(degree: int, coef0: float = 1.0) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def poly(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (torch.dot(x, y) + coef0) ** degree
    return poly


def _sigmoid(alpha: float, coef0: float = 0.0) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def sigmoid(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.tanh(alpha * torch.dot(x, y) + coef0)
    return sigmoid


class QuantumKernelMethod(nn.Module):
    """
    Classical kernel factory exposing RBF, polynomial, and sigmoid kernels.

    Parameters
    ----------
    kernel : str, optional
        One of ``'rbf'``, ``'poly'``, ``'sigmoid'``; defaults to ``'rbf'``.
    gamma : float, optional
        Control parameter for RBF and sigmoid kernels.
    degree : int, optional
        Degree for polynomial kernel; ignored otherwise.
    coef0 : float, optional
        Independent term in polynomial and sigmoid kernels; ignored otherwise.

    Notes
    -----
    The class acts as a thin wrapper around a callable kernel function
    that accepts two 1‑D tensors and returns a scalar similarity.
    """

    _kernel_registry = {
        "rbf": _default_rbf,
        "poly": _polynomial,
        "sigmoid": _sigmoid,
    }

    def __init__(
        self,
        kernel: str = "rbf",
        *,
        gamma: float = 1.0,
        degree: int = 3,
        coef0: float = 1.0,
    ) -> None:
        super().__init__()
        if kernel not in self._kernel_registry:
            raise ValueError(f"Unsupported kernel: {kernel}")
        self.kernel_name = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_fn = self._kernel_registry[kernel](
            gamma if kernel == "rbf" else None,
            degree if kernel == "poly" else None,
            coef0 if kernel in ("poly", "sigmoid") else None,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between two 1‑D tensors.

        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of the same shape.

        Returns
        -------
        torch.Tensor
            Scalar similarity.
        """
        return self.kernel_fn(x, y)

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
        *,
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Compute the Gram matrix between datasets ``a`` and ``b``.

        Supports optional batching to keep memory usage in check.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Sequences of 1‑D tensors.
        batch_size : int, optional
            Number of rows to compute in each batch.

        Returns
        -------
        np.ndarray
            Gram matrix of shape (len(a), len(b)).
        """
        n_a = len(a)
        n_b = len(b)
        gram = np.empty((n_a, n_b), dtype=np.float64)
        for i in range(0, n_a, batch_size):
            end_i = min(i + batch_size, n_a)
            batch_x = torch.stack(a[i:end_i])
            for j in range(0, n_b, batch_size):
                end_j = min(j + batch_size, n_b)
                batch_y = torch.stack(b[j:end_j])
                # Broadcasting: (batch_x, 1) vs (1, batch_y)
                sims = torch.exp(
                    -self.gamma
                    * torch.sum(
                        (batch_x.unsqueeze(1) - batch_y.unsqueeze(0)) ** 2,
                        dim=2,
                        keepdim=True,
                    )
                ).squeeze(-1)
                gram[i:end_i, j:end_j] = sims.cpu().numpy()
        return gram


__all__ = ["QuantumKernelMethod"]
