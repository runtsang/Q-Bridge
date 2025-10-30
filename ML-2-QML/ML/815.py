"""Enhanced classical RBF kernel with learnable bandwidth and kernel ridge regression."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

class LearnableRBF(nn.Module):
    """Learnable RBF kernel with a trainable bandwidth parameter."""
    def __init__(self, init_gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel matrix between x and y.
        x, y: tensors of shape (n_samples, n_features)
        Returns: kernel matrix of shape (n_samples, n_samples)
        """
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        sq_dist = torch.sum(diff ** 2, dim=-1)
        return torch.exp(-self.gamma * sq_dist)

class QuantumKernelMethod(nn.Module):
    """Wrapper that provides a learnable RBF kernel and a simple kernel ridge regression."""
    def __init__(self, init_gamma: float = 1.0, alpha: float = 1.0) -> None:
        super().__init__()
        self.kernel = LearnableRBF(init_gamma)
        self.alpha = alpha
        self.train_X = None
        self.train_y = None
        self.train_K_inv = None

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Fit kernel ridge regression."""
        K = self.kernel(X, X)
        n = K.shape[0]
        K_reg = K + self.alpha * torch.eye(n, device=K.device, dtype=K.dtype)
        self.train_K_inv = torch.inverse(K_reg)
        self.train_X = X
        self.train_y = y

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict on new data."""
        if self.train_K_inv is None:
            raise RuntimeError("Model has not been fitted yet.")
        K_test = self.kernel(X, self.train_X)
        return torch.matmul(K_test, torch.matmul(self.train_K_inv, self.train_y))

    def kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper to compute kernel matrix."""
        return self.kernel(X, Y)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], init_gamma: float = 1.0) -> np.ndarray:
    """Compute kernel matrix between two lists of tensors using LearnableRBF."""
    model = QuantumKernelMethod(init_gamma=init_gamma)
    X = torch.stack(a)
    Y = torch.stack(b)
    return model.kernel_matrix(X, Y).detach().cpu().numpy()

__all__ = ["LearnableRBF", "QuantumKernelMethod", "kernel_matrix"]
