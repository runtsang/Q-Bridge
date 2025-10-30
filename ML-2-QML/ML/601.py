"""Enhanced classical RBF kernel with parameter learning and SVM integration."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC

__all__ = ["QuantumKernelMethod"]


class QuantumKernelMethod(nn.Module):
    """A learnable RBF kernel that can be used with a downstream SVM."""
    def __init__(self, gamma: float | None = None, gamma_init: float = 1.0) -> None:
        super().__init__()
        if gamma is None:
            # Learnable gamma
            self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))
        else:
            self.gamma = torch.tensor(gamma, dtype=torch.float32)
        self.svm = None
        self.X_train = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> np.ndarray:
        """Compute the Gram matrix between X and Y."""
        K = torch.exp(-self.gamma * torch.cdist(X, Y, p=2).pow(2))
        return K.detach().cpu().numpy()

    def fit(self, X: np.ndarray, y: np.ndarray, C: float = 1.0, kernel: str = "precomputed") -> None:
        """Fit a support vector machine using the learned kernel."""
        self.X_train = torch.tensor(X, dtype=torch.float32)
        K = self.kernel_matrix(self.X_train, self.X_train)
        self.svm = SVC(C=C, kernel=kernel)
        self.svm.fit(K, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for new data using the trained SVM."""
        if self.svm is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_tensor = torch.tensor(X, dtype=torch.float32)
        K = self.kernel_matrix(X_tensor, self.X_train)
        return self.svm.predict(K)
