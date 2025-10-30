"""Enhanced classical RBF kernel and classifier with automatic bandwidth selection."""

from __future__ import annotations

from typing import Sequence, Tuple, List

import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.svm import SVC

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix", "KernClassifier"]

class KernalAnsatz(nn.Module):
    """RBF kernel function with optional gamma parameter."""
    def __init__(self, gamma: float | None = None) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Convenience wrapper that automatically selects gamma if None."""
    def __init__(self, gamma: float | None = None) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[np.ndarray], b: Sequence[np.ndarray], gamma: float | None = None) -> np.ndarray:
    """Compute the Gram matrix between two datasets with optional automatic gamma via the median trick."""
    a = np.vstack(a)
    b = np.vstack(b)
    if gamma is None:
        # median trick
        dists = np.linalg.norm(a[:, None] - b, axis=2) ** 2
        median = np.median(dists)
        gamma = 1.0 / (2 * median)
    kernel = Kernel(gamma)
    return np.array([[kernel(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)).item() for y in b] for x in a])

class KernClassifier(BaseEstimator, ClassifierMixin):
    """Multi‑class classifier based on a pre‑computed kernel matrix."""
    def __init__(self, gamma: float | None = None, C: float = 1.0):
        self.gamma = gamma
        self.C = C
        self.svc_ = None
        self.X_ = None

    def fit(self, X: Sequence[np.ndarray], y: Sequence[int]) -> "KernClassifier":
        X, y = check_X_y(X, y)
        self.X_ = X
        gram = kernel_matrix(X, X, self.gamma)
        self.svc_ = SVC(C=self.C, kernel="precomputed")
        self.svc_.fit(gram, y)
        return self

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        if self.svc_ is None or self.X_ is None:
            raise ValueError("The model has not been fitted yet.")
        gram = kernel_matrix(X, self.X_, self.gamma)
        return self.svc_.predict(gram)
