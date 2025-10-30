"""Enhanced kernel utilities with adaptive bandwidth, neural, and hybrid kernels.

This module implements several kernel variants that are useful for
the original class‑only RBF implementation while adding
the *information‑gain*‐to‑the‑data‑driven set‑point.
"""

from __future__ import annotations

import typing as t

import numpy as np
import torch
import torch.nn as nn


class KernelBase:
    """Base class for all kernels."""
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AdaptiveRBFKernel(KernelBase):
    """RBF kernel with bandwidth learned from data."""
    def __init__(self, gamma: float | None = None, median_heuristic: bool = True):
        self.gamma = gamma
        self.median_heuristic = median_heuristic

    def fit(self, data: torch.Tensor):
        if self.median_heuristic:
            # Compute median of pairwise distances
            dists = torch.cdist(data, data)
            median = torch.median(dists)
            self.gamma = 1.0 / (2 * median ** 2 + 1e-8)
        elif self.gamma is None:
            raise ValueError("gamma must be specified if not using median heuristic")

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class NeuralNetworkKernel(KernelBase):
    """Kernel defined by a neural network feature map."""
    def __init__(self, hidden_dims: t.Sequence[int] = (64, 32), out_dim: int = 16):
        layers = []
        dims = list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z_x = self.net(x)
        z_y = self.net(y)
        return torch.sum(z_x * z_y, dim=-1, keepdim=True)


class HybridKernel(KernelBase):
    """Combines a classical RBF kernel with a quantum kernel."""
    def __init__(self, rbf_gamma: float = 1.0, quantum_kernel=None):
        self.rbf = AdaptiveRBFKernel(gamma=rbf_gamma, median_heuristic=False)
        self.quantum_kernel = quantum_kernel

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        rbf_val = self.rbf(x, y)
        if self.quantum_kernel is not None:
            q_val = self.quantum_kernel(x, y)
            return rbf_val + q_val
        return rbf_val


class KernelFactory:
    """Factory for constructing kernels."""
    @staticmethod
    def create(kernel_type: str, **kwargs) -> KernelBase:
        mapping = {
            "rbf": AdaptiveRBFKernel,
            "nn": NeuralNetworkKernel,
            "hybrid": HybridKernel,
        }
        if kernel_type not in mapping:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        return mapping[kernel_type](**kwargs)


class Kernel:
    """Unified kernel interface used by downstream algorithms."""
    def __init__(self, kernel_type: str = "rbf", **kwargs):
        self._kernel = KernelFactory.create(kernel_type, **kwargs)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._kernel(x, y)

    def fit(self, data: torch.Tensor):
        """Fit the underlying kernel if it implements fit."""
        if hasattr(self._kernel, "fit"):
            self._kernel.fit(data)


def kernel_matrix(a: t.Sequence[torch.Tensor], b: t.Sequence[torch.Tensor], kernel_type: str = "rbf", **kwargs) -> np.ndarray:
    """Compute Gram matrix between datasets ``a`` and ``b``."""
    kernel = Kernel(kernel_type, **kwargs)
    if hasattr(kernel._kernel, "fit"):
        kernel.fit(torch.stack(a))
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["Kernel", "AdaptiveRBFKernel", "NeuralNetworkKernel", "HybridKernel", "KernelFactory", "kernel_matrix"]
