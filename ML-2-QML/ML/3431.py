"""Hybrid kernel and estimator combining classical RBF and analytic quantum kernel."""

import numpy as np
import torch
from torch import nn
from typing import Sequence

class HybridQuantumKernelEstimator(nn.Module):
    """Hybrid kernel that blends a classical RBF with an analytic quantum kernel.

    The kernel is defined as:
        K(x, y) = w * Q(x, y) + (1-w) * RBF(x, y)

    Where Q is the overlap of a 4‑qubit Ry encoding circuit, which can be
    evaluated analytically as a product of cos((x_i - y_i)/2).
    The RBF kernel is the usual exp(-γ||x-y||²).
    """
    def __init__(self, gamma: float = 1.0, quantum_weight: float = 0.5, n_wires: int = 4):
        super().__init__()
        self.gamma = gamma
        self.quantum_weight = quantum_weight
        self.n_wires = n_wires

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def _quantum(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        cos_term = torch.cos(diff / 2.0)
        return torch.prod(cos_term, dim=-1, keepdim=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        rbf_val = self._rbf(x, y)
        q_val = self._quantum(x, y)
        return self.quantum_weight * q_val + (1.0 - self.quantum_weight) * rbf_val

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix between two sets of vectors."""
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

    def fit(self, X: torch.Tensor, y: torch.Tensor, alpha: float = 1e-3) -> None:
        """Simple kernel ridge regression fit."""
        K = self.kernel_matrix(X, X)
        self.coef_ = torch.linalg.solve(torch.tensor(K) + alpha * torch.eye(len(X)),
                                        y.unsqueeze(-1)).squeeze()

    def predict(self, X: torch.Tensor, X_train: torch.Tensor) -> torch.Tensor:
        K_test = self.kernel_matrix(X, X_train)
        return K_test @ self.coef_

class EstimatorNN(nn.Module):
    """Simple feed‑forward regressor mirroring the classical part of EstimatorQNN."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)

__all__ = ["HybridQuantumKernelEstimator", "EstimatorNN"]
