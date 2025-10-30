"""Hybrid kernel method combining classical RBF and a quantum kernel."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV


class ClassicalRBF(nn.Module):
    """Pure RBF kernel."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class QuantumKernel(nn.Module):
    """Placeholder quantum kernel – inner‑product of a one‑hot encoding."""

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.abs((x @ y.t()).diag().unsqueeze(-1))


class QuantumKernelMethod(nn.Module):
    """
    Hybrid kernel that mixes classical RBF and a quantum kernel.
    Provides a kernel‑ridge regression interface with hyper‑parameter tuning.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        alpha: float = 1.0,
        mix: float = 0.5,
        use_quantum: bool = True,
    ) -> None:
        super().__init__()
        self.rbf = ClassicalRBF(gamma)
        self.quantum = QuantumKernel() if use_quantum else None
        self.mix = mix
        self.alpha = alpha
        self.krr = KernelRidge(alpha=self.alpha)

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute hybrid kernel value."""
        k_rbf = self.rbf(x, y)
        if self.quantum is not None:
            k_q = self.quantum(x, y)
            return self.mix * k_rbf + (1.0 - self.mix) * k_q
        return k_rbf

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Fit kernel ridge regression on training data."""
        K = self.kernel(X, X).detach().cpu().numpy()
        self.krr.fit(K, y.cpu().numpy())

    def predict(self, X: torch.Tensor, X_train: torch.Tensor) -> torch.Tensor:
        """Predict on new data."""
        K_test = self.kernel(X, X_train).detach().cpu().numpy()
        return torch.tensor(self.krr.predict(K_test), dtype=X.dtype, device=X.device)

    def kernel_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """Return Gram matrix between two sequences of tensors."""
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])


__all__ = ["QuantumKernelMethod", "ClassicalRBF", "QuantumKernel"]
