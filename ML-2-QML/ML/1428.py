"""Enhanced classical RBF kernel module with scaling and vectorised kernel matrix.

The module introduces:
- Optional feature scaling (mean/std) for numerical stability.
- A ``fit`` method that pre‑computes scaling statistics.
- A vectorised implementation of the Gram matrix via ``torch.cdist``.
- A convenience wrapper ``QuantumKernelMethod`` that mirrors the original API.

The public names ``KernalAnsatz``, ``Kernel`` and ``kernel_matrix`` are kept for backward
compatibility, while ``QuantumKernelMethod`` is the new, user‑facing class.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class KernalAnsatz(nn.Module):
    """Enhanced RBF kernel ansatz with optional scaling."""

    def __init__(self, gamma: float = 1.0, scale: bool = True) -> None:
        super().__init__()
        self.gamma = gamma
        self.scale = scale
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "KernalAnsatz":
        """Compute mean and std for feature scaling."""
        if self.scale:
            self._mean = X.mean(axis=0, keepdims=True)
            self._std = X.std(axis=0, keepdims=True) + 1e-8
        return self

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        if self.scale and self._mean is not None and self._std is not None:
            return (X - self._mean) / self._std
        return X

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Wrapper that exposes a sklearn‑style interface."""

    def __init__(self, gamma: float = 1.0, scale: bool = True) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma, scale)

    def fit(self, X: np.ndarray) -> "Kernel":
        self.ansatz.fit(X)
        return self

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)

    def kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        X = self.ansatz._preprocess(X)
        Y = self.ansatz._preprocess(Y)
        X_t = torch.tensor(X, dtype=torch.float32)
        Y_t = torch.tensor(Y, dtype=torch.float32)
        dists = torch.cdist(X_t, Y_t, p=2)
        return torch.exp(-self.ansatz.gamma * dists**2).numpy()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


class QuantumKernelMethod:
    """Convenience wrapper that mirrors the original API."""

    def __init__(self, gamma: float = 1.0, scale: bool = True) -> None:
        self.kernel = Kernel(gamma, scale)

    def fit(self, X: np.ndarray) -> "QuantumKernelMethod":
        self.kernel.fit(X)
        return self

    def kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.kernel.kernel_matrix(X, Y)


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix", "QuantumKernelMethod"]
