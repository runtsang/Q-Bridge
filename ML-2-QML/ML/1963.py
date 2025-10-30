"""Enhanced classical RBF kernel with automatic bandwidth selection and minibatch evaluation."""
from __future__ import annotations

from typing import Sequence, Iterable, Optional

import numpy as np
import torch
from torch import nn
from sklearn.kernel_ridge import KernelRidge

class KernalAnsatz(nn.Module):
    """Immutable RBF kernel component, kept for API compatibility."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class QuantumKernelMethod:
    """
    Wrapper around an RBF kernel that automatically tunes the bandwidth (γ)
    using cross‑validation and evaluates the kernel matrix in minibatches
    for large datasets.

    Parameters
    ----------
    gamma : float, optional
        Explicit bandwidth. If None, γ is tuned via cross‑validation on
        the supplied training data.
    gamma_grid : Sequence[float], optional
        Search space for γ when tuning. Defaults to a logarithmic grid.
    cv_folds : int, default 5
        Number of folds for cross‑validation.
    batch_size : int, default 200
        Number of samples per batch when computing the Gram matrix.
    """
    def __init__(
        self,
        gamma: Optional[float] = None,
        gamma_grid: Optional[Sequence[float]] = None,
        cv_folds: int = 5,
        batch_size: int = 200,
    ) -> None:
        self.gamma = gamma
        self.gamma_grid = gamma_grid or [1e-3, 1e-2, 1e-1, 1, 10]
        self.cv_folds = cv_folds
        self.batch_size = batch_size

    def fit(self, X: Iterable[np.ndarray], y: Optional[Iterable[np.ndarray]] = None) -> "QuantumKernelMethod":
        """
        Fit the kernel to the data by selecting the best γ via cross‑validation.
        """
        X_np = np.asarray(list(X))
        y_np = np.asarray(list(y)) if y is not None else np.zeros(X_np.shape[0])

        best_gamma = None
        best_score = -np.inf
        for g in self.gamma_grid:
            ridge = KernelRidge(alpha=1.0, kernel="rbf", gamma=g)
            scores = []
            n = X_np.shape[0]
            fold_size = max(1, n // self.cv_folds)
            for i in range(self.cv_folds):
                start, end = i * fold_size, i * fold_size + fold_size
                X_train = np.concatenate([X_np[:start], X_np[end:]])
                y_train = np.concatenate([y_np[:start], y_np[end:]])
                X_val = X_np[start:end]
                y_val = y_np[start:end]
                ridge.fit(X_train, y_train)
                scores.append(ridge.score(X_val, y_val))
            avg = np.mean(scores)
            if avg > best_score:
                best_score = avg
                best_gamma = g

        self.gamma = best_gamma
        return self

    def kernel_matrix(self, X: Sequence[np.ndarray], Y: Sequence[np.ndarray]) -> np.ndarray:
        """
        Compute the Gram matrix between X and Y using the tuned γ.
        The computation is performed in mini‑batches to keep memory usage
        low for large datasets.
        """
        if self.gamma is None:
            raise RuntimeError("The kernel must be fitted before calling kernel_matrix.")
        X_np = np.asarray(X)
        Y_np = np.asarray(Y)
        n, m = X_np.shape[0], Y_np.shape[0]
        K = np.empty((n, m), dtype=np.float64)

        for i in range(0, n, self.batch_size):
            X_batch = torch.tensor(X_np[i : i + self.batch_size], dtype=torch.float32)
            for j in range(0, m, self.batch_size):
                Y_batch = torch.tensor(Y_np[j : j + self.batch_size], dtype=torch.float32)
                diff = X_batch.unsqueeze(1) - Y_batch.unsqueeze(0)
                val = torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1))
                K[i : i + X_batch.shape[0], j : j + Y_batch.shape[0]] = val.numpy()
        return K

    def __call__(self, X: Sequence[np.ndarray], Y: Sequence[np.ndarray]) -> np.ndarray:
        return self.kernel_matrix(X, Y)

__all__ = ["QuantumKernelMethod", "KernalAnsatz"]
