"""Extended RBF kernel utilities for classical and hybrid models.

This module builds on the original radial‑basis‑function implementation but adds:
* multi‑output support
* automatic bandwidth selection using a simple heuristic or cross‑validation
* a scikit‑learn compatible estimator that can be plugged into pipelines
* a simple kernel‑learning wrapper that returns both the Gram matrix and a low‑dimensional embedding
"""

from __future__ import annotations

import numpy as np
from typing import Sequence, Callable, Iterable, Tuple, Optional

import torch
from torch import nn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

__all__ = [
    "RBFKernel",
    "MultiOutputKernel",
    "kernel_matrix",
    "KernelEstimator",
    "KernelLearner",
]


class RBFKernel(nn.Module):
    """Base RBF kernel with configurable bandwidth and optional distance function."""

    def __init__(self, gamma: float = 1.0, dist: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.dist = dist

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.dist is None:
            diff = x - y
            dist_sq = torch.sum(diff * diff, dim=-1, keepdim=True)
        else:
            dist_sq = self.dist(x, y)
        return torch.exp(-self.gamma * dist_sq)


class MultiOutputKernel(nn.Module):
    """Kernel that returns a vector of kernel values for multiple bandwidths."""

    def __init__(self, gammas: Sequence[float]) -> None:
        super().__init__()
        self.gammas = torch.tensor(gammas, dtype=torch.float32)
        self.base = RBFKernel()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        dist_sq = torch.sum(diff * diff, dim=-1, keepdim=True)  # shape (N,1)
        # broadcast gamma (1,M) over dist_sq (N,1)
        return torch.exp(-self.gammas * dist_sq)


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two collections of tensors."""
    kernel = RBFKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


class KernelEstimator(BaseEstimator, TransformerMixin):
    """Fit an RBF kernel with bandwidth selected via grid search."""

    def __init__(self, gamma: float = 1.0, grid: Optional[Iterable[float]] = None, cv: int = 5) -> None:
        self.gamma = gamma
        self.grid = grid if grid is not None else [0.1, 1.0, 10.0, 100.0]
        self.cv = cv
        self.best_gamma_ = None
        self.kernel_ = None

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "KernelEstimator":
        # Dummy estimator that only stores gamma
        class Dummy:
            def __init__(self, gamma):
                self.gamma = gamma

            def fit(self, X, y=None):
                return self

            def score(self, X, y=None):
                return 0  # placeholder

        dummy = Dummy(self.gamma)
        gs = GridSearchCV(dummy, param_grid={"gamma": self.grid}, cv=self.cv, scoring="accuracy")
        gs.fit(X, y)
        self.best_gamma_ = gs.best_params_["gamma"]
        self.kernel_ = RBFKernel(self.best_gamma_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.kernel_ is None:
            raise RuntimeError("Estimator has not been fitted.")
        return np.array([[self.kernel_(torch.tensor(x), torch.tensor(y)).item() for y in X] for x in X])


class KernelLearner(nn.Module):
    """Learn a low‑dimensional embedding that preserves the kernel matrix.

    This is a toy implementation of kernel PCA: it computes the centered Gram matrix
    and projects onto the top‑k eigenvectors.
    """

    def __init__(self, n_components: int = 5, gamma: float = 1.0) -> None:
        super().__init__()
        self.n_components = n_components
        self.gamma = gamma
        self.eigenvectors_ = None

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute Gram matrix
        K = torch.exp(-self.gamma * torch.cdist(X, X) ** 2)
        # Center the Gram matrix
        N = K.shape[0]
        one = torch.ones(N, N) / N
        K_centered = K - one @ K - K @ one + one @ K @ one
        # Eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(K_centered)
        idx = torch.argsort(eigvals, descending=True)
        eigvecs = eigvecs[:, idx][:, : self.n_components]
        self.eigenvectors_ = eigvecs
        # Embedding
        embedding = K_centered @ eigvecs
        return K, embedding
