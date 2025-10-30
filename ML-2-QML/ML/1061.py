"""Hybrid kernel module with learnable ensemble and RBF weighting."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence, List, Tuple


class KernelBase(nn.Module):
    """Base class for kernel implementations with a common interface."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return a scalar kernel value for a pair of inputs."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class RBFKernel(KernelBase):
    """RBF kernel with a trainable weight and set‑to‑zero‑like bias."""
    def __init__(self, gamma: float = 1.0, weight: float = 1.0):
        super().__init__(gamma)
        self.weight = nn.Parameter(torch.tensor(weight, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return self.weight * torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class EnsembleKernel(nn.Module):
    """Weighted sum of multiple kernels with learnable coefficients."""
    def __init__(self, kernels: List[KernelBase]):
        super().__init__()
        self.kernels = nn.ModuleList(kernels)
        self.coeffs = nn.Parameter(torch.ones(len(kernels)) / len(kernels))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        coeffs = torch.softmax(self.coeffs, dim=0)
        out = torch.stack([k(x, y) for k in self.kernels], dim=0)
        return torch.sum(coeffs.unsqueeze(-1) * out, dim=0)


class HybridKernel(nn.Module):
    """Convenience wrapper that trains an ensemble of RBF kernels."""
    def __init__(self, gammas: List[float], weights: List[float] = None):
        super().__init__()
        kernels = [RBFKernel(gamma, w) for gamma, w in zip(gammas, weights or [1.0]*len(gammas))]
        self.ensemble = EnsembleKernel(kernels)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ensemble(x, y)

    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 100, lr: float = 1e-3) -> None:
        """Simple gradient‑descent training of kernel weights."""
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            preds = self.forward(X, X).diag()
            loss = torch.mean((preds - y)**2)
            loss.backward()
            opt.step()

    def predict(self, X: torch.Tensor, X_train: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
        """Kernel ridge regression prediction using the learned kernel."""
        K = self.forward(X_train, X_train)
        alpha = torch.linalg.solve(K + 1e-5 * torch.eye(K.shape[0]), y_train)
        K_test = self.forward(X, X_train)
        return K_test @ alpha


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], kernel: KernelBase) -> np.ndarray:
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = [
    "KernelBase",
    "RBFKernel",
    "EnsembleKernel",
    "HybridKernel",
    "kernel_matrix",
]
