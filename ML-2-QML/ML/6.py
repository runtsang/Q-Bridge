"""QuantumKernel: versatile classical kernel implementation."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence, Callable, Optional

class QuantumKernel(nn.Module):
    """
    Classical kernel module supporting RBF, polynomial, and linear kernels.
    Designed to be dropped into scikit‑learn pipelines.
    """

    def __init__(
        self,
        kernel_type: str = "rbf",
        gamma: float = 1.0,
        degree: int = 3,
        coef0: float = 1.0,
        custom_func: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        """
        Parameters
        ----------
        kernel_type : {"rbf", "poly", "linear", "custom"}
            The kernel to use. ``custom`` requires ``custom_func``.
        gamma : float, optional
            Kernel coefficient for rbf and poly kernels.
        degree : int, optional
            Degree for polynomial kernel.
        coef0 : float, optional
            Independent term in the polynomial kernel.
        custom_func : callable, optional
            A user supplied callable returning a scalar kernel value.
        """
        super().__init__()
        self.kernel_type = kernel_type.lower()
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.custom_func = custom_func

        allowed = {"rbf", "poly", "linear", "custom"}
        if self.kernel_type not in allowed:
            raise ValueError(f"Unsupported kernel_type: {kernel_type}")

        if self.kernel_type == "custom" and custom_func is None:
            raise ValueError("custom_func must be supplied for kernel_type='custom'")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel value between two 1‑D tensors.
        """
        if self.kernel_type == "rbf":
            diff = x - y
            return torch.exp(-self.gamma * torch.sum(diff * diff))
        if self.kernel_type == "poly":
            return (self.gamma * torch.dot(x, y) + self.coef0) ** self.degree
        if self.kernel_type == "linear":
            return torch.dot(x, y)
        # custom
        return self.custom_func(x, y)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute the Gram matrix between two lists of feature vectors.
        """
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

    def __repr__(self) -> str:
        return f"<QuantumKernel type={self.kernel_type}>"

__all__ = ["QuantumKernel"]
