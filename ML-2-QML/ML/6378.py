"""Hybrid classical kernel model with a learnable RBF bandwidth and an SVM classifier.

The :class:`QuantumKernelMethod__gen192` class:
* Wraps a learnable RBF kernel (`LearnableRBFKernel`) whose bandwidth `gamma`
  is a trainable torch parameter.
* Uses scikit‑learn's ``SVC`` with a pre‑computed kernel to perform
  classification.
* Provides helpers to compute the Gram matrix and to fit/predict.

This design keeps the original API (``kernel_matrix``) while adding a
trainable hyper‑parameter and a ready‑to‑use pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from sklearn import svm

__all__ = ["QuantumKernelMethod__gen192", "LearnableRBFKernel", "Kernel"]

class LearnableRBFKernel(nn.Module):
    """RBF kernel with a learnable bandwidth `gamma`."""

    def __init__(self, gamma_init: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Convenience wrapper that keeps the original API."""

    def __init__(self, gamma_init: float = 1.0) -> None:
        super().__init__()
        self.ansatz = LearnableRBFKernel(gamma_init)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: list[torch.Tensor], b: list[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix from two sequences of tensors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class QuantumKernelMethod__gen192(nn.Module):
    """Hybrid kernel‑based learning module.

    Parameters
    ----------
    gamma_init : float, optional
        Initial bandwidth of the RBF kernel.
    """

    def __init__(self, gamma_init: float = 1.0) -> None:
        super().__init__()
        self.kernel = Kernel(gamma_init)
        self.svm = svm.SVC(kernel="precomputed")
        self.trained = False

    def fit(self, X: list[torch.Tensor], y: np.ndarray) -> None:
        """Fit the SVM on the Gram matrix of ``X``."""
        G = self.kernel_matrix(X, X)
        self.svm.fit(G, y)
        self.trained = True

    def predict(self, X: list[torch.Tensor], X_train: list[torch.Tensor]) -> np.ndarray:
        """Predict labels for ``X`` using the trained SVM."""
        if not self.trained:
            raise RuntimeError("Model must be fitted before prediction.")
        G = self.kernel_matrix(X, X_train)
        return self.svm.predict(G)

    @staticmethod
    def kernel_matrix(a: list[torch.Tensor], b: list[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix between two datasets."""
        kernel = Kernel()
        return np.array([[kernel(x, y).item() for y in b] for x in a])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compatibility forward method – returns kernel value."""
        return self.kernel(x, y)
