"""Extended classical kernel module with GPU support and multi‑kernel options."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from sklearn.metrics import pairwise_kernels
from typing import Sequence, Literal, Union

class QuantumKernelMethod(nn.Module):
    """
    Classical kernel wrapper that supports RBF, linear, polynomial and
    custom callable kernels.  The class is fully differentiable via
    PyTorch autograd and can run on CUDA if available.

    Parameters
    ----------
    kernel : str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Kernel type.  Supported strings: `"rbf"`, `"linear"`, `"poly"`.
        Custom callables must accept two 1‑D tensors and return a
        scalar similarity.
    gamma : float, optional
        RBF bandwidth; used only when ``kernel="rbf"``.
    degree : int, optional
        Polynomial degree; used only when ``kernel="poly"``.
    coef0 : float, optional
        Coefficient for polynomial kernel.
    """
    def __init__(
        self,
        kernel: Union[str, callable] = "rbf",
        gamma: float = 1.0,
        degree: int = 3,
        coef0: float = 0.0,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1))

    def _poly(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (torch.dot(x, y) + self.coef0) ** self.degree

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Return similarity between two 1‑D tensors.
        """
        if isinstance(self.kernel, str):
            if self.kernel == "rbf":
                return self._rbf(x, y)
            elif self.kernel == "linear":
                return torch.dot(x, y)
            elif self.kernel == "poly":
                return self._poly(x, y)
            else:
                raise ValueError(f"Unsupported kernel: {self.kernel}")
        else:
            return self.kernel(x, y)

    def kernel_matrix(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor] | None = None,
    ) -> np.ndarray:
        """
        Compute Gram matrix between X and Y.  X, Y can be NumPy arrays
        or torch tensors.  The result is returned as a NumPy array.
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if Y is None:
            Y = X
        elif isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).float()

        # Batch pairwise evaluation
        X_exp = X.unsqueeze(1)  # (N, 1, d)
        Y_exp = Y.unsqueeze(0)  # (1, M, d)
        diff = X_exp - Y_exp  # (N, M, d)

        if isinstance(self.kernel, str):
            if self.kernel == "rbf":
                K = torch.exp(-self.gamma * torch.sum(diff ** 2, dim=-1))
            elif self.kernel == "linear":
                K = torch.matmul(X, Y.T)
            elif self.kernel == "poly":
                K = (torch.matmul(X, Y.T) + self.coef0) ** self.degree
            else:
                raise ValueError(f"Unsupported kernel: {self.kernel}")
        else:
            # custom callable: vectorised via broadcasting
            K = torch.stack([self.kernel(x, y) for x in X for y in Y]).reshape(X.shape[0], Y.shape[0])

        return K.cpu().numpy()

__all__ = ["QuantumKernelMethod"]
