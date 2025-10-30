# Classical kernel utilities with sklearn compatibility and hyper-parameter tuning.

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score

__all__ = ["QuantumKernelMethod", "kernel_matrix", "RBFKernel"]

class RBFKernel(nn.Module):
    'Radial Basis Function kernel with optional gamma scaling.'
    def __init__(self, gamma: float | None = None):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float) -> np.ndarray:
    'Compute Gram matrix for RBF kernel.'
    k = RBFKernel(gamma)
    return np.array([[k(x, y).item() for y in b] for x in a])

class QuantumKernelMethod(BaseEstimator, RegressorMixin):
    'Unified interface for classical RBF kernel regression.'
    def __init__(self, gamma: float = 1.0, alpha: float = 1.0):
        self.gamma = gamma
        self.alpha = alpha

    def fit(self, X, y):
        self.X_train_ = torch.tensor(X, dtype=torch.float32)
        self.y_train_ = torch.tensor(y, dtype=torch.float32)
        self.K_train_ = kernel_matrix(self.X_train_, self.X_train_, self.gamma)
        self.w_ = torch.linalg.solve(
            torch.tensor(self.K_train_, dtype=torch.float32) + self.alpha * torch.eye(len(self.X_train_)),
            self.y_train_
        )
        return self

    def predict(self, X):
        X_test = torch.tensor(X, dtype=torch.float32)
        K_test = kernel_matrix(X_test, self.X_train_, self.gamma)
        return (K_test @ self.w_.numpy()).flatten()

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
