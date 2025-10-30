"""Classical kernel methods with support for RBF and polynomial kernels.

This module extends the original radial basis function implementation by
adding vectorized kernel matrix computation, optional GPU support via PyTorch,
and a clean API that mirrors the quantum counterpart.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Iterable, Sequence, Union

class QuantumKernelMethod:
    """Classical kernel evaluator.

    Parameters
    ----------
    kernel_type : str, default='rbf'
        The kernel to use. Options are ``'rbf'`` and ``'poly'``.
    gamma : float, default=1.0
        RBF kernel coefficient.
    degree : int, default=3
        Degree for the polynomial kernel.

    Notes
    -----
    The class implements a callable interface ``__call__(x, y)`` that
    returns the kernel value between two vectors, and a ``kernel_matrix`` method
    that accepts two sequences of samples and returns the Gram matrix.
    """

    def __init__(self, kernel_type: str = "rbf", gamma: float = 1.0, degree: int = 3) -> None:
        self.kernel_type = kernel_type.lower()
        self.gamma = gamma
        self.degree = degree
        if self.kernel_type not in {"rbf", "poly"}:
            raise ValueError(f"Unsupported kernel_type: {kernel_type}")

    def _rbf(self, x: np.ndarray, y: np.ndarray) -> float:
        diff = x - y
        return float(np.exp(-self.gamma * np.sum(diff ** 2)))

    def _poly(self, x: np.ndarray, y: np.ndarray) -> float:
        return float((np.dot(x, y) + 1) ** self.degree)

    def __call__(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> float:
        """Compute the kernel value between two vectors."""
        x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=float)
        y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y, dtype=float)

        if self.kernel_type == "rbf":
            return self._rbf(x_np, y_np)
        else:
            return self._poly(x_np, y_np)

    def kernel_matrix(self, a: Sequence[Union[np.ndarray, torch.Tensor]],
                      b: Sequence[Union[np.ndarray, torch.Tensor]]) -> np.ndarray:
        """Compute the Gram matrix between two collections of samples.

        Parameters
        ----------
        a, b : sequences of vectors
            Input data. Elements may be NumPy arrays or torch tensors.
        """
        # Convert to NumPy arrays for vectorized computation
        a_np = np.asarray([x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in a], dtype=float)
        b_np = np.asarray([y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y for y in b], dtype=float)

        if self.kernel_type == "rbf":
            # Efficient pairwise squared distances
            aa = np.sum(a_np ** 2, axis=1).reshape(-1, 1)
            bb = np.sum(b_np ** 2, axis=1).reshape(1, -1)
            sq_dist = aa + bb - 2 * a_np @ b_np.T
            return np.exp(-self.gamma * sq_dist)
        else:  # polynomial
            return (a_np @ b_np.T + 1) ** self.degree

__all__ = ["QuantumKernelMethod"]
