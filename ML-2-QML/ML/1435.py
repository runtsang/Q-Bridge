"""Enhanced classical RBF kernel with bandwidth auto‑tuning and hybrid support."""

from __future__ import annotations

from typing import Sequence, Callable, Optional

import numpy as np
import torch
from torch import nn
from sklearn.metrics import pairwise_distances

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "HybridKernel",
    "KernelFactory",
    "auto_bandwidth",
]


class KernalAnsatz(nn.Module):
    """RBF kernel function with optional bias and gamma auto‑tuning."""

    def __init__(self, gamma: float = 1.0, bias: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.bias = bias

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True)) + self.bias


class Kernel(nn.Module):
    """Wrapper that provides a callable kernel and Gram matrix computation."""

    def __init__(self, gamma: float = 1.0, bias: float = 0.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma, bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

    def gram_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """Compute full Gram matrix for dataset X."""
        n = X.shape[0]
        K = torch.empty((n, n), dtype=X.dtype, device=X.device)
        for i in range(n):
            K[i] = self.forward(X[i], X)
        return K


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix using efficient pairwise distance."""
    X = torch.stack(a)
    Y = torch.stack(b)
    dists = pairwise_distances(X.numpy(), Y.numpy(), metric="sqeuclidean")
    K = np.exp(-gamma * dists)
    return K


def auto_bandwidth(X: torch.Tensor, percentile: float = 90.0) -> float:
    """Estimate bandwidth gamma as inverse of median of pairwise distances."""
    dists = pairwise_distances(X.numpy(), metric="sqeuclidean")
    median = np.median(dists)
    return 1.0 / (2 * median) if median > 0 else 1.0


class HybridKernel(nn.Module):
    """Blend classical RBF and quantum kernel with weight alpha."""

    def __init__(self, classical_kernel: nn.Module, quantum_kernel: nn.Module, alpha: float = 0.5) -> None:
        super().__init__()
        self.classical = classical_kernel
        self.quantum = quantum_kernel
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.classical(x, y)
        q = self.quantum(x, y)
        return self.alpha * c + (1 - self.alpha) * q


class KernelFactory:
    """Factory to instantiate kernels based on configuration dict."""

    @staticmethod
    def create(cfg: dict) -> nn.Module:
        ktype = cfg.get("type", "rbf")
        if ktype == "rbf":
            gamma = cfg.get("gamma", 1.0)
            return Kernel(gamma=gamma)
        elif ktype == "hybrid":
            gamma = cfg.get("gamma", 1.0)
            alpha = cfg.get("alpha", 0.5)
            qkernel = cfg["quantum"]
            return HybridKernel(Kernel(gamma=gamma), qkernel, alpha=alpha)
        else:
            raise ValueError(f"Unsupported kernel type: {ktype}")
