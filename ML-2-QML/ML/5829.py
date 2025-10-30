"""Classical radial basis function kernel with trainable hyper‑parameter and GP support."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

__all__ = ["QuantumKernelMethod", "kernel_matrix"]

class QuantumKernelMethod(nn.Module):
    """Trainable RBF kernel with Gaussian Process regression support."""
    def __init__(self, gamma: float = 1.0, noise: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        self.noise = noise

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel matrix between two batches of samples."""
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        sqdist = torch.sum(diff * diff, dim=-1)
        return torch.exp(-self.gamma * sqdist)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix between two sequences of tensors."""
        a_t = torch.stack(a)
        b_t = torch.stack(b)
        K = self.forward(a_t, b_t)
        return K.detach().cpu().numpy()

    def train_gamma(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 200):
        """Optimize gamma by minimizing the negative log marginal likelihood."""
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        optimizer = Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            optimizer.zero_grad()
            K = self.forward(X_t, X_t) + self.noise * torch.eye(len(X_t))
            L = torch.cholesky(K)
            alpha = torch.cholesky_solve(y_t.unsqueeze(1), L)
            nll = 0.5 * y_t @ alpha.squeeze() + torch.sum(torch.log(torch.diag(L))) + 0.5 * len(X_t) * np.log(2 * np.pi)
            nll.backward()
            optimizer.step()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix using a fresh instance of QuantumKernelMethod."""
    kernel = QuantumKernelMethod(gamma)
    return kernel.kernel_matrix(a, b)
