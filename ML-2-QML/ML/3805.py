"""Hybrid kernel ridge regression combining classical RBF and trainable quantum kernels.

The implementation is intentionally lightweight:
* Classical RBF kernel is computed with Torch tensors.
* Quantum kernel is driven by a reusable :class:`QuantumKernelVar` (defined in the QML module).
* The regressor solves the regularised linear system in closed form.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Classical kernel utilities
# --------------------------------------------------------------------------- #
class ClassicalRBF(nn.Module):
    """Radial‑basis function kernel implemented in Torch."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return kernel value for pair (x, y)."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# --------------------------------------------------------------------------- #
# Helper to compute kernel matrices
# --------------------------------------------------------------------------- #
def _kernel_matrix(kernel: nn.Module, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix K_{i,j} = kernel(a_i, b_j)."""
    n, d = a.shape
    m, _ = b.shape
    # Expand for broadcasting
    a_exp = a.unsqueeze(1).expand(n, m, d)
    b_exp = b.unsqueeze(0).expand(n, m, d)
    return kernel(a_exp.reshape(-1, d), b_exp.reshape(-1, d)).reshape(n, m)

# --------------------------------------------------------------------------- #
# Hybrid kernel regressor
# --------------------------------------------------------------------------- #
class HybridKernelRegressor(nn.Module):
    """
    Kernel‑ridge regressor using a sum of a classical RBF kernel and a
    variational quantum kernel.  The quantum kernel is supplied via an
    external :class:`QuantumKernelVar` implementation.

    Parameters
    ----------
    gamma : float
        Width of the classical RBF kernel.
    lambda_reg : float
        Regularisation coefficient.
    quantum_kernel_cls : type
        Class that implements the quantum kernel (must inherit from
        :class:`torchquantum.QuantumModule` and expose a ``forward``
        method returning a scalar kernel value).
    """

    def __init__(self,
                 gamma: float = 1.0,
                 lambda_reg: float = 1e-5,
                 quantum_kernel_cls: type = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.classical_kernel = ClassicalRBF(gamma)
        if quantum_kernel_cls is None:
            raise ValueError("`quantum_kernel_cls` must be provided.")
        self.quantum_kernel = quantum_kernel_cls()

        # Will be set during fit
        self.train_X: torch.Tensor | None = None
        self.alpha: torch.Tensor | None = None

    # --------------------------------------------------------------------- #
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Fit the ridge regression model.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, n_features)
        y : torch.Tensor of shape (n_samples, 1)
        """
        self.train_X = X
        Kc = _kernel_matrix(self.classical_kernel, X, X)
        Kq = _kernel_matrix(self.quantum_kernel, X, X)
        K = Kc + Kq

        # Solve (K + λI) α = y
        n = K.shape[0]
        A = K + self.lambda_reg * torch.eye(n, device=K.device, dtype=K.dtype)
        self.alpha = torch.linalg.solve(A, y)

    # --------------------------------------------------------------------- #
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict outputs for new samples.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples_new, n_features)

        Returns
        -------
        torch.Tensor of shape (n_samples_new, 1)
        """
        if self.train_X is None or self.alpha is None:
            raise RuntimeError("Model has not been fitted yet.")
        Kc = _kernel_matrix(self.classical_kernel, X, self.train_X)
        Kq = _kernel_matrix(self.quantum_kernel, X, self.train_X)
        K = Kc + Kq
        return K @ self.alpha

    # --------------------------------------------------------------------- #
    def kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Return combined kernel matrix between X and Y."""
        Kc = _kernel_matrix(self.classical_kernel, X, Y)
        Kq = _kernel_matrix(self.quantum_kernel, X, Y)
        return Kc + Kq

__all__ = ["HybridKernelRegressor", "ClassicalRBF", "HybridKernelRegressor"]
