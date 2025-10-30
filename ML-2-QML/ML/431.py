"""QuantumKernelMethod: classical kernel implementations with caching and feature scaling.

This module provides a unified :class:`QuantumKernelMethod` interface that supports
RBF, polynomial and linear kernels. Feature scaling is optional and the
implementation is fully vectorised using PyTorch. A lightweight cache
stores previously computed kernel values for pairs of tensors to accelerate
repeated queries.
"""

import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from typing import Sequence, Tuple, Dict, Optional

class QuantumKernelMethod(nn.Module):
    """A classical kernel function with optional feature scaling and caching.

    Parameters
    ----------
    kernel : str
        Name of the kernel.  Supported: 'rbf', 'poly', 'linear'.
    gamma : float, default=1.0
        Parameter for the RBF kernel.
    degree : int, default=3
        Degree for the polynomial kernel.
    coef0 : float, default=1.0
        Independent term in polynomial kernel.
    feature_scale : bool, default=False
        If ``True`` data are scaled to zero mean and unit variance before
        kernel evaluation.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: float = 1.0,
        degree: int = 3,
        coef0: float = 1.0,
        feature_scale: bool = False,
    ) -> None:
        super().__init__()
        self.kernel = kernel.lower()
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.feature_scale = feature_scale
        self._scaler = StandardScaler() if feature_scale else None
        self._cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_scale:
            mean = x.mean()
            std = x.std()
            return (x - mean) / (std + 1e-8)
        return x

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        dist_sq = torch.sum(diff * diff, dim=-1, keepdim=True)
        return torch.exp(-self.gamma * dist_sq)

    def _poly(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dot = torch.sum(x * y, dim=-1, keepdim=True)
        return (self.gamma * dot + self.coef0) ** self.degree

    def _linear(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sum(x * y, dim=-1, keepdim=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value for a single pair of vectors.

        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of identical shape.

        Returns
        -------
        torch.Tensor
            A scalar tensor containing the kernel value.
        """
        x = self._scale(x)
        y = self._scale(y)
        key = (id(x), id(y))
        if key in self._cache:
            return self._cache[key]
        if self.kernel == "rbf":
            k = self._rbf(x, y)
        elif self.kernel == "poly":
            k = self._poly(x, y)
        elif self.kernel == "linear":
            k = self._linear(x, y)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel}")
        self._cache[key] = k
        return k

    def kernel_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """Compute the Gram matrix between two collections of vectors.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Sequences of 1‑D tensors.

        Returns
        -------
        np.ndarray
            2‑D array of shape ``(len(a), len(b))`` containing kernel values.
        """
        mat = torch.empty((len(a), len(b)), dtype=torch.float32)
        for i, xi in enumerate(a):
            for j, yj in enumerate(b):
                mat[i, j] = self.forward(xi, yj)
        return mat.cpu().numpy()


__all__ = ["QuantumKernelMethod"]
