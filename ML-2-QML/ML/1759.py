"""Enhanced radial basis function kernel with GPU acceleration and sklearn integration."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

class KernalAnsatz(nn.Module):
    """Parameterised RBF kernel with support for multiple kernel types."""

    def __init__(self, gamma: float = 1.0, kernel_type: str = "rbf",
                 degree: int = 3) -> None:
        super().__init__()
        self.gamma = gamma
        self.kernel_type = kernel_type.lower()
        self.degree = degree

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the kernel value between two 1‑D tensors."""
        x = x.view(1, -1)
        y = y.view(1, -1)
        diff = x - y
        if self.kernel_type == "rbf":
            return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))
        elif self.kernel_type == "poly":
            return (1 + self.gamma * torch.sum(x * y, dim=-1, keepdim=True)) ** self.degree
        elif self.kernel_type == "sigmoid":
            return torch.tanh(self.gamma * torch.sum(x * y, dim=-1, keepdim=True) + 1)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

class Kernel(nn.Module, BaseEstimator, TransformerMixin):
    """Wrapper for :class:`KernalAnsatz` that exposes a sklearn‑style API."""

    def __init__(self, gamma: float = 1.0, kernel_type: str = "rbf",
                 degree: int = 3, use_gpu: bool = False) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma, kernel_type, degree)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            self.ansatz.cuda()

    def _check_inputs(self, X):
        X = check_array(X, accept_sparse="csr", dtype=float, ensure_2d=True)
        return torch.tensor(X, dtype=torch.float32,
                            device="cuda" if self.use_gpu else "cpu")

    def transform(self, X):
        """Return the kernel matrix K(X, X) as a 2‑D array."""
        X = self._check_inputs(X)
        n_samples = X.shape[0]
        # Broadcast x and y
        X_exp = X.unsqueeze(1)  # (n,1,d)
        Y_exp = X.unsqueeze(0)  # (1,n,d)
        K = self.ansatz(X_exp.reshape(-1, X.shape[1]),
                        Y_exp.reshape(-1, Y_exp.shape[2]))
        K = K.reshape(n_samples, n_samples)
        return K.cpu().numpy()

    def fit(self, X, y=None):
        """No‑op to satisfy sklearn estimator interface."""
        return self

    def decision_function(self, X):
        """Compatibility shim for sklearn SVMs.

        Returns the diagonal of the kernel matrix, i.e. K(X, X).
        """
        return self.transform(X)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                 gamma: float = 1.0) -> np.ndarray:
    """Compute pairwise RBF kernel matrix for two lists of 1‑D tensors."""
    kernel = Kernel(gamma=gamma, kernel_type="rbf", use_gpu=False)
    A = torch.stack(a)
    B = torch.stack(b)
    # Broadcast pairs
    K = kernel.ansatz(A.unsqueeze(1).reshape(-1, A.shape[1]),
                      B.unsqueeze(0).reshape(-1, B.shape[1]))
    return K.cpu().numpy().reshape(len(a), len(b))

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
