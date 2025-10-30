"""Enhanced classical kernel module with multiple kernel types and hyperparameter optimisation."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from typing import Sequence, Callable, Optional


class Kernel(nn.Module):
    """Kernel module supporting RBF, Polynomial, and Matern kernels.

    Parameters
    ----------
    kernel_type : str, optional
        One of ``'rbf'``, ``'poly'`` or ``'matern'``. Default is ``'rbf'``.
    gamma : float, optional
        RBF or Matern length‑scale. Default is 1.0.
    degree : int, optional
        Polynomial kernel degree. Default is 3.
    coef0 : float, optional
        Polynomial kernel coefficient. Default is 1.0.
    nu : float, optional
        Matern kernel shape parameter (only used if kernel_type =='matern'). Default is 1.5.
    """

    def __init__(
        self,
        kernel_type: str = "rbf",
        gamma: float = 1.0,
        degree: int = 3,
        coef0: float = 1.0,
        nu: float = 1.5,
    ) -> None:
        super().__init__()
        if kernel_type not in ("rbf", "poly", "matern"):
            raise ValueError(f"Unsupported kernel_type {kernel_type!r}")
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.nu = nu

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Compute kernel value between two batches."""
        if self.kernel_type == "rbf":
            return self._rbf(x, y)
        elif self.kernel_type == "poly":
            return self._poly(x, y)
        else:
            return self._matern(x, y)

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Radial Basis Function kernel."""
        diff = x[:, None, :] - y[None, :, :]
        sq_norm = torch.sum(diff * diff, dim=-1)
        return torch.exp(-self.gamma * sq_norm)

    def _poly(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Polynomial kernel."""
        prod = torch.matmul(x, y.t())
        return (self.gamma * prod + self.coef0) ** self.degree

    def _matern(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Matern kernel with nu=1.5 or 2.5."""
        from math import sqrt

        diff = x[:, None, :] - y[None, :, :]
        r = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-10)
        if self.nu == 1.5:
            return (
                (1 + sqrt(3) * r / self.gamma)
                * torch.exp(-sqrt(3) * r / self.gamma)
            )
        elif self.nu == 2.5:
            return (
                (1 + sqrt(5) * r / self.gamma + 5 * r * r / (3 * self.gamma * self.gamma))
                * torch.exp(-sqrt(5) * r / self.gamma)
            )
        else:
            raise ValueError("Unsupported nu for Matern kernel")

    def kernel_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """Compute Gram matrix between two sequences of tensors."""
        a = torch.stack(a)
        b = torch.stack(b)
        return self(a, b).detach().cpu().numpy()

    def fit_hyperparameters(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        lr: float = 0.01,
        epochs: int = 200,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> None:
        """Gradient‑based optimisation of the hyperparameter ``gamma`` for RBF and Matern kernels.

        The loss is the mean squared error between the kernel matrix and the target similarity
        matrix ``y`` (e.g. labels encoded as a similarity matrix). Only ``gamma`` is updated.
        """
        if self.kernel_type not in ("rbf", "matern"):
            raise RuntimeError(
                "Hyperparameter optimisation only implemented for RBF and Matern kernels."
            )

        gamma_param = torch.tensor(self.gamma, requires_grad=True)
        optimizer = Adam([gamma_param], lr=lr)

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            k = self._rbf(X, X) if self.kernel_type == "rbf" else self._matern(X, X)
            loss = torch.mean((k - y) ** 2)
            loss.backward()
            optimizer.step()
            self.gamma = gamma_param.item()
            if callback:
                callback(epoch, loss.item())


__all__ = ["Kernel"]
