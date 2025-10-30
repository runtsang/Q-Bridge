#!/usr/bin/env python
"""Quantum kernel method – classical implementation.

Provides a flexible RBF/linear kernel with vectorised Gram matrix
computation, hyperparameter handling and a convenient API for
integration with scikit‑learn.
"""

from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
import torch
from torch import nn

class QuantumKernelMethod(nn.Module):
    """
    Classical kernel wrapper supporting RBF and linear kernels.

    Parameters
    ----------
    kernel_type : {'rbf', 'linear'}, default 'rbf'
        Type of kernel to evaluate.
    gamma : float, default 1.0
        RBF width parameter; ignored for linear kernel.
    n_features : int | None, default None
        Expected dimensionality of input data.  If provided, the module
        will validate incoming tensors for shape consistency.
    """

    def __init__(self, kernel_type: str = "rbf", gamma: float = 1.0, n_features: Optional[int] = None) -> None:
        super().__init__()
        if kernel_type not in {"rbf", "linear"}:
            raise ValueError(f"Unsupported kernel_type {kernel_type!r}")
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.n_features = n_features

    def _check_input(self, x: torch.Tensor) -> None:
        if self.n_features is not None and x.shape[-1]!= self.n_features:
            raise ValueError(f"Expected feature size {self.n_features}, got {x.shape[-1]}")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel value between two 1‑D tensors.

        The tensors are flattened and reshaped to (1, -1) before
        evaluation to keep the interface identical to the quantum
        counterpart.
        """
        x = x.view(1, -1)
        y = y.view(1, -1)
        self._check_input(x)
        self._check_input(y)

        if self.kernel_type == "rbf":
            diff = x - y
            return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))
        else:  # linear
            return torch.dot(x.squeeze(), y.squeeze()).unsqueeze(0)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Optional[Sequence[torch.Tensor]] = None) -> np.ndarray:
        """
        Compute the Gram matrix between two collections of tensors.

        Parameters
        ----------
        a : Sequence[torch.Tensor]
            First collection of 1‑D tensors.
        b : Sequence[torch.Tensor] | None, default None
            Second collection.  If None, the matrix is computed for
            ``a`` against itself.

        Returns
        -------
        np.ndarray
            2‑D Gram matrix of shape (len(a), len(b)).
        """
        if b is None:
            b = a
        kernel = self
        return np.array([[kernel(x, y).item() for y in b] for x in a])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kernel_type={self.kernel_type!r}, gamma={self.gamma:.3g})"

__all__ = ["QuantumKernelMethod"]
