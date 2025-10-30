"""Hybrid kernel module with trainable bandwidth and optional feature selection."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class RBFKernel(nn.Module):
    """Classical RBF kernel with a learnable bandwidth γ."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute exp(-γ * ||x - y||²) for two batches of vectors.
        x: (n, d)
        y: (m, d)
        returns: (n, m)
        """
        diff = x.unsqueeze(1) - y.unsqueeze(0)          # (n, m, d)
        sq_norm = torch.sum(diff**2, dim=-1)             # (n, m)
        return torch.exp(-self.gamma * sq_norm)

def kernel_matrix(a: torch.Tensor, b: torch.Tensor, gamma: float | None = None) -> np.ndarray:
    """
    Compute the Gram matrix between two batches of vectors.
    a: (n, d)
    b: (m, d)
    """
    if gamma is None:
        gamma = 1.0
    diff = a.unsqueeze(1) - b.unsqueeze(0)
    sq_norm = torch.sum(diff**2, dim=-1)
    K = torch.exp(-gamma * sq_norm)
    return K.detach().cpu().numpy()

def train_svm_with_kernel(
    X: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[svm.SVC, float]:
    """
    Train an SVM with a pre‑computed kernel matrix and return the model and test accuracy.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)

    K_train = kernel_matrix(X_train_t, X_train_t, gamma=gamma)
    K_test  = kernel_matrix(X_test_t, X_train_t, gamma=gamma)

    clf = svm.SVC(kernel="precomputed")
    clf.fit(K_train, y_train)
    acc = accuracy_score(y_test, clf.predict(K_test))
    return clf, acc

__all__ = ["RBFKernel", "kernel_matrix", "train_svm_with_kernel"]
