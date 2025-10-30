"""Enhanced classical RBF kernel utilities with hyper‑parameter search and SVM wrapper."""

from __future__ import annotations

from typing import Sequence, Iterable, Union, Tuple
import numpy as np
import torch
from torch import nn
from sklearn import svm
from sklearn.model_selection import GridSearchCV

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "SVMKernelClassifier",
    "HyperParameterSearch",
]

class KernalAnsatz(nn.Module):
    """Radial basis function kernel with support for per‑feature gamma values."""
    def __init__(self, gamma: Union[float, Iterable[float]] = 1.0) -> None:
        super().__init__()
        if isinstance(gamma, (list, tuple, np.ndarray)):
            self.gamma = torch.tensor(gamma, dtype=torch.float32)
        else:
            self.gamma = torch.tensor([gamma], dtype=torch.float32)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Compute squared Euclidean distance with per‑feature weighting
        diff = x - y
        weighted_sq = torch.sum((diff * self.gamma) ** 2, dim=-1, keepdim=True)
        return torch.exp(-weighted_sq)

class Kernel(nn.Module):
    """Kernel module that wraps :class:`KernalAnsatz`."""
    def __init__(self, gamma: Union[float, Iterable[float]] = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: Union[float, Iterable[float]] = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two datasets."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class SVMKernelClassifier:
    """Support‑vector‑machine that uses a pre‑computed kernel matrix."""
    def __init__(self, gamma: Union[float, Iterable[float]] = 1.0, cv: int = 5):
        self.gamma = gamma
        self.cv = cv
        self.model = None

    def fit(self, X: Sequence[torch.Tensor], y: Sequence[int]) -> None:
        X = torch.stack(list(X))
        y = np.array(y)
        kernel_mat = kernel_matrix(X, X, self.gamma)
        self.model = svm.SVC(kernel='precomputed')
        self.model.fit(kernel_mat, y)

    def predict(self, X_train: Sequence[torch.Tensor], X_test: Sequence[torch.Tensor]) -> np.ndarray:
        X_train = torch.stack(list(X_train))
        X_test = torch.stack(list(X_test))
        kernel_mat = kernel_matrix(X_test, X_train, self.gamma)
        return self.model.predict(kernel_mat)

class HyperParameterSearch:
    """Utility to perform a grid search over gamma values for the RBF kernel."""
    def __init__(self, gamma_grid: Iterable[float]):
        self.gamma_grid = gamma_grid

    def search(self, X: Sequence[torch.Tensor], y: Sequence[int], cv: int = 5):
        X = torch.stack(list(X))
        y = np.array(y)
        best_gamma = None
        best_score = -np.inf
        for gamma in self.gamma_grid:
            kernel_mat = kernel_matrix(X, X, gamma)
            clf = svm.SVC(kernel='precomputed')
            scores = []
            # Simple leave‑one‑out cross‑validation
            for i in range(len(X)):
                mask = np.ones(len(X), dtype=bool)
                mask[i] = False
                clf.fit(kernel_mat[mask][:, mask], y[mask])
                scores.append(clf.score(kernel_mat[mask][i], y[i:i+1]))
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_gamma = gamma
        return best_gamma, best_score
