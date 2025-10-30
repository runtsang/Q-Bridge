"""Classical kernel method with adaptive RBF, multi‑kernel fusion and hyper‑parameter tuning."""

from __future__ import annotations

from typing import Sequence, List, Optional

import numpy as np
import torch
from torch import nn


class KernalAnsatz(nn.Module):
    """RBF kernel ansatz with adaptive gamma."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """RBF kernel module that wraps :class:`KernalAnsatz`."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y).squeeze()


class QuantumKernelMethod(nn.Module):
    """Hybrid kernel method that supports RBF, linear and polynomial kernels with fusion."""
    def __init__(self,
                 kernel_type: str = "rbf",
                 gamma: Optional[float] = None,
                 degree: int = 3,
                 coef0: float = 1.0,
                 lambda_reg: float = 1e-3) -> None:
        super().__init__()
        self.kernel_type = kernel_type.lower()
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.lambda_reg = lambda_reg

        if self.kernel_type == "rbf":
            if self.gamma is None:
                self.gamma = 1.0
            self.kernel_fn = self._rbf_kernel
        elif self.kernel_type == "linear":
            self.kernel_fn = self._linear_kernel
        elif self.kernel_type == "poly":
            self.kernel_fn = self._poly_kernel
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def _linear_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sum(x * y, dim=-1, keepdim=True)

    def _poly_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (torch.sum(x * y, dim=-1, keepdim=True) * self.coef0 + 1) ** self.degree

    def fit_gamma(self, X: torch.Tensor) -> None:
        """Fit gamma using median heuristic on pairwise distances."""
        if self.kernel_type!= "rbf":
            return
        dists = torch.cdist(X, X, p=2)
        mask = torch.triu(torch.ones_like(dists, dtype=bool), diagonal=1)
        median = torch.median(dists[mask])
        self.gamma = 1.0 / (2.0 * median ** 2)

    def kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix between X and Y using selected kernel."""
        n, d = X.shape
        m, _ = Y.shape
        X_exp = X.unsqueeze(1).expand(n, m, d)
        Y_exp = Y.unsqueeze(0).expand(n, m, d)
        K = self.kernel_fn(X_exp.reshape(-1, d), Y_exp.reshape(-1, d))
        return K.reshape(n, m)

    def fuse_kernels(self, kernels: List["QuantumKernelMethod"]) -> "QuantumKernelMethod":
        """Fuse multiple kernels by summing their Gram matrices."""
        class FusedKernel(QuantumKernelMethod):
            def __init__(self, kernels):
                super().__init__(kernel_type="rbf")  # dummy
                self.kernels = kernels

            def kernel_matrix(self, X, Y):
                K = 0.0
                for k in self.kernels:
                    K += k.kernel_matrix(X, Y)
                return K

        return FusedKernel(kernels)

    def to_numpy(self, X: torch.Tensor) -> np.ndarray:
        return self.kernel_matrix(X, X).detach().cpu().numpy()

    def __repr__(self):
        return f"QuantumKernelMethod(kernel_type={self.kernel_type}, gamma={self.gamma}, degree={self.degree})"


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["KernalAnsatz", "Kernel", "QuantumKernelMethod", "kernel_matrix"]
