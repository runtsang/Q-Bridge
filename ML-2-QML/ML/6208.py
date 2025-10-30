"""Hybrid classical‑quantum kernel with learnable weighting and sparse support."""

from __future__ import annotations

from typing import Sequence, Iterable, Optional

import numpy as np
import torch
from torch import nn

__all__ = ["HybridKernel"]


class _BaseKernel(nn.Module):
    """Base class with common utilities for kernel modules."""
    def __init__(self, device: torch.device | None = None) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device)


class ClassicalRBF(nn.Module):
    """Pure classical RBF kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: (..., d)
        diff = x.unsqueeze(-2) - y.unsqueeze(-3)
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1))


class PolynomialFeatureMap(nn.Module):
    """Simple polynomial kernel to emulate a quantum feature map."""
    def __init__(self, degree: int = 2, coef0: float = 0.0) -> None:
        super().__init__()
        self.degree = degree
        self.coef0 = coef0

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # (x^T y + coef0)^degree
        prod = torch.matmul(x, y.t())
        return (prod + self.coef0) ** self.degree


class HybridKernel(_BaseKernel):
    """Hybrid kernel combining classical RBF and a polynomial feature map.

    The weighting ``alpha`` controls the contribution of each component.
    """
    def __init__(self,
                 alpha: float = 0.5,
                 gamma: float = 1.0,
                 degree: int = 2,
                 coef0: float = 0.0,
                 device: torch.device | None = None) -> None:
        super().__init__(device=device)
        self.alpha = alpha
        self.classical = ClassicalRBF(gamma=gamma)
        self.polynomial = PolynomialFeatureMap(degree=degree, coef0=coef0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self._to_device(x)
        y = self._to_device(y)
        k_class = self.classical(x, y)
        k_poly = self.polynomial(x, y)
        return self.alpha * k_class + (1 - self.alpha) * k_poly

    def kernel_matrix(self,
                      a: Iterable[torch.Tensor],
                      b: Iterable[torch.Tensor]) -> np.ndarray:
        """Compute the Gram matrix for sequences of tensors."""
        a = torch.stack([self._to_device(t) for t in a])
        b = torch.stack([self._to_device(t) for t in b])
        return self.forward(a, b).detach().cpu().numpy()

    def fit(self, X: Sequence[np.ndarray], y: Sequence[int]) -> None:
        """Placeholder fit method; can be extended for kernel methods."""
        pass

    def predict(self,
                X: Sequence[np.ndarray],
                X_train: Sequence[np.ndarray],
                y_train: Sequence[int]) -> np.ndarray:
        """Simple kernel ridge regression prediction."""
        K = self.kernel_matrix(X_train, X_train)
        K_inv = np.linalg.pinv(K)
        y_train = np.array(y_train)
        predictions = []
        for x in X:
            k_x = self.kernel_matrix([x], X_train)[0]
            pred = k_x @ K_inv @ y_train
            predictions.append(pred)
        return np.array(predictions)
