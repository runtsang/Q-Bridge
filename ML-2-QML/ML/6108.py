"""Enhanced classical RBF kernel with SVM regression."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, RegressorMixin


class KernalAnsatz(nn.Module):
    """RBF kernel ansatz with a trainable gamma parameter."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Kernel module that wraps :class:`KernalAnsatz`."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two batches of data."""
    kernel = Kernel(gamma)
    return np.array([[kernel(torch.tensor(x), torch.tensor(y)).item() for y in b] for x in a])


class QuantumKernelMethod(BaseEstimator, RegressorMixin):
    """Hybrid estimator that uses a classical RBF kernel and an SVM for regression."""
    def __init__(self, gamma: float = 1.0, C: float = 1.0, fit_intercept: bool = True):
        self.gamma = gamma
        self.C = C
        self.fit_intercept = fit_intercept
        self._svm: SVC | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelMethod":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        K = kernel_matrix([torch.tensor(x) for x in X], [torch.tensor(x) for x in X], self.gamma)
        self._svm = SVC(kernel="precomputed", C=self.C, fit_intercept=self.fit_intercept)
        self._svm.fit(K, y)
        self.X_train_ = X
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._svm is None:
            raise RuntimeError("The model must be fitted before calling predict.")
        X = np.asarray(X, dtype=np.float32)
        K = kernel_matrix([torch.tensor(x) for x in X], [torch.tensor(x) for x in self.X_train_], self.gamma)
        return self._svm.predict(K)

    @property
    def gamma_(self) -> float:
        """Return the current gamma value used by the kernel."""
        return float(self.gamma)


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix", "QuantumKernelMethod"]
