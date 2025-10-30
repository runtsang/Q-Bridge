"""Enhanced classical RBF kernel with hyper‑parameter search and multi‑scale support."""

from __future__ import annotations

from typing import Sequence, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils import check_array, check_X_y

__all__ = ["KernalKernel", "Kernel", "KernelMatrix", "KernelRegressor", "KernelClassifier"]


def _check_gammas(gamma: float | Iterable[float]) -> Tuple[float,...]:
    """Return a tuple of distinct gamma values for multi‑scale RBF."""
    if isinstance(gamma, (list, tuple)):
        if not all(isinstance(g, (float, int)) for g in gamma):
            raise ValueError("All gamma values must be numeric")
        return tuple(g for g in gamma if g > 0)
    else:
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        return (float(gamma),)


class KernalKernel(nn.Module):
    """Placeholder maintaining compatibility with the quantum interface."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """RBF kernel module that wraps :class:`KernalKernel`."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalKernel(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def KernelMatrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float | Iterable[float] = 1.0) -> np.ndarray:
    """Compute a multi‑scale Gram matrix between two collections of tensors."""
    gammas = _check_gammas(gamma)
    kernels = [Kernel(g) for g in gammas]
    return np.sum([np.array([[k(x, y).item() for y in b] for x in a]) for k in kernels], axis=0)


class KernelRegressor(BaseEstimator, RegressorMixin):
    """A scikit‑learn regressor that uses a quantum‑style kernel matrix."""

    def __init__(self, gamma: float | Iterable[float] = 1.0, alpha: float = 1e-3):
        self.gamma = gamma
        self.alpha = alpha

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
        self.kernel_ = KernelMatrix(X, X, self.gamma)
        self.coef_ = np.linalg.solve(self.kernel_ + self.alpha * np.eye(len(X)), y)
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        K = KernelMatrix(X, self.kernel_, self.gamma)
        return K @ self.coef_


class KernelClassifier(BaseEstimator, ClassifierMixin):
    """A scikit‑learn classifier that uses a quantum‑style kernel matrix."""

    def __init__(self, gamma: float | Iterable[float] = 1.0, C: float = 1.0):
        self.gamma = gamma
        self.C = C

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
        self.kernel_ = KernelMatrix(X, X, self.gamma)
        # simple linear SVM on the kernel space
        from sklearn.svm import LinearSVC
        self.svm_ = LinearSVC(C=self.C)
        self.svm_.fit(self.kernel_, y)
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        K = KernelMatrix(X, self.kernel_, self.gamma)
        return self.svm_.predict(K)
