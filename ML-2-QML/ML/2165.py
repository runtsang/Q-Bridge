"""Classical kernel methods with flexible kernel types and GPU support."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class QuantumKernel(nn.Module):
    """Flexible classical kernel module supporting RBF and polynomial kernels.

    Parameters
    ----------
    kernel_type : str, optional
        Kernel type: ``"rbf"`` or ``"poly"``. Default is ``"rbf"``.
    gamma : float, optional
        Scaling factor for the RBF kernel. Ignored for polynomial kernel.
    degree : int, optional
        Degree of the polynomial kernel. Ignored for RBF kernel.
    """

    def __init__(
        self,
        kernel_type: str = "rbf",
        gamma: float = 1.0,
        degree: int = 3,
    ) -> None:
        super().__init__()
        self.kernel_type = kernel_type.lower()
        self.gamma = gamma
        self.degree = degree
        if self.kernel_type not in {"rbf", "poly"}:
            raise ValueError(f"Unsupported kernel_type {kernel_type!r}")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute kernel value between two batches of vectors.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(m, d)`` or ``(d,)``.
        y : torch.Tensor
            Tensor of shape ``(n, d)`` or ``(d,)``.

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape ``(m, n)``.
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if y.ndim == 1:
            y = y.unsqueeze(0)

        diff = x[:, None, :] - y[None, :, :]
        if self.kernel_type == "rbf":
            return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1))
        else:
            dot = torch.matmul(x, y.t())
            return (dot + 1.0) ** self.degree


def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    kernel_type: str = "rbf",
    gamma: float = 1.0,
    degree: int = 3,
) -> np.ndarray:
    """Compute the Gram matrix between two collections of feature vectors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of tensors with identical dimensionality.
    kernel_type, gamma, degree : see :class:`QuantumKernel`.

    Returns
    -------
    np.ndarray
        Gram matrix of shape ``(len(a), len(b))``.
    """
    kernel = QuantumKernel(kernel_type, gamma, degree)
    A = torch.stack(a)
    B = torch.stack(b)
    return kernel(A, B).cpu().numpy()


__all__ = ["QuantumKernel", "kernel_matrix"]
