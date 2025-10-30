"""Enhanced classical kernel module with optional hybrid support and feature selection."""

from __future__ import annotations

from typing import Sequence, List, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    "RBFKernel",
    "FeatureSelector",
    "HybridKernel",
    "QuantumKernelMethod",
    "kernel_matrix",
]

class RBFKernel(nn.Module):
    """Classical RBF kernel with optional gamma scaling."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class FeatureSelector(nn.Module):
    """Simple feature selector based on variance threshold."""

    def __init__(self, threshold: float = 0.0) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        var = torch.var(X, dim=0)
        mask = var > self.threshold
        return X[:, mask]

class HybridKernel(nn.Module):
    """Hybrid kernel combining RBF and a placeholder quantum kernel."""

    def __init__(self, gamma: float = 1.0, use_quantum: bool = False) -> None:
        super().__init__()
        self.rbf = RBFKernel(gamma)
        self.use_quantum = use_quantum
        # Placeholder for quantum kernel; in practice this would be replaced
        # by a call to a quantum backend or a precomputed kernel matrix.
        self.quantum_scale = 0.5 if use_quantum else 0.0

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        rbf_val = self.rbf(x, y)
        # In a real hybrid scenario, we would compute a quantum kernel value here.
        quantum_val = torch.ones_like(rbf_val) * self.quantum_scale
        return rbf_val + quantum_val

class QuantumKernelMethod(nn.Module):
    """Main model that integrates feature selection, a hybrid kernel, and an MLP head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        gamma: float = 1.0,
        use_quantum: bool = False,
        feature_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.selector = FeatureSelector(feature_threshold)
        self.kernel = HybridKernel(gamma, use_quantum)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute kernel matrix between X and y, then pass through MLP."""
        X_sel = self.selector(X)
        y_sel = self.selector(y)
        # Compute pairwise kernel matrix
        K = self.kernel(X_sel, y_sel)
        # Flatten kernel output for MLP
        K_flat = K.reshape(K.size(0), -1)
        return self.mlp(K_flat)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Utility to compute Gram matrix between two sequences of tensors."""
    kernel = RBFKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])
