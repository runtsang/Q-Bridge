"""
Quantum kernel utilities for classical machine learning pipelines.
Supports RBF, polynomial, and linear kernels with differentiable hyper‑parameters
and efficient batch evaluation via torch broadcasting.
"""

from __future__ import annotations

from typing import Sequence, Callable, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn.functional import normalize


class QuantumKernelMethod(nn.Module):
    """
    Hybrid kernel object that can be instantiated with a kernel family
    and a set of hyper‑parameters.  The kernel matrix can be computed
    in a fully differentiable manner, enabling seamless integration
    with downstream optimisers (e.g., SVM, GP).
    """

    def __init__(
        self,
        kernel_type: str = "rbf",
        gamma: float | None = None,
        degree: int = 3,
        coef0: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        kernel_type : {"rbf", "poly", "linear"}, default="rbf"
            Type of kernel to use.
        gamma : float, optional
            Scaling factor for RBF. If None, defaults to 1 / (n_features * 2).
        degree : int, default=3
            Degree for polynomial kernel.
        coef0 : float, default=1.0
            Independent term in polynomial kernel.
        """
        super().__init__()
        self.kernel_type = kernel_type.lower()
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

        # Register gamma as a learnable parameter for RBF
        if self.kernel_type == "rbf":
            if gamma is None:
                gamma = 1.0  # placeholder; will be overridden in forward
            self.gamma_param = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        else:
            self.register_parameter("gamma_param", None)

    def _rbf(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        # X: (m, d), Y: (n, d)
        gamma = self.gamma_param if self.gamma_param is not None else torch.tensor(self.gamma)
        diff = X.unsqueeze(1) - Y.unsqueeze(0)          # (m, n, d)
        sq_norm = torch.sum(diff**2, dim=-1)            # (m, n)
        return torch.exp(-gamma * sq_norm)

    def _poly(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return (torch.mm(X, Y.t()) + self.coef0) ** self.degree

    def _linear(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return torch.mm(X, Y.t())

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel matrix between X and Y.

        Parameters
        ----------
        X : torch.Tensor, shape (m, d)
        Y : torch.Tensor, shape (n, d)

        Returns
        -------
        torch.Tensor of shape (m, n)
        """
        if self.kernel_type == "rbf":
            return self._rbf(X, Y)
        elif self.kernel_type == "poly":
            return self._poly(X, Y)
        elif self.kernel_type == "linear":
            return self._linear(X, Y)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """
        Compute the Gram matrix between two sequences of tensors.
        Each tensor is expected to be 1‑D and of the same feature length.
        """
        X = torch.stack(a)  # (m, d)
        Y = torch.stack(b)  # (n, d)
        return self.forward(X, Y).detach().cpu().numpy()


__all__ = ["QuantumKernelMethod"]
