"""Hybrid kernel framework with classical and quantum support.

Defines a unified API for classical kernel methods with multi‑kernel fusion.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence, Callable

class QuantumKernelMethod(nn.Module):
    """
    Classical kernel module that supports multiple kernels (RBF, Matern, etc.) and
    allows weighted fusion of kernels.  The API mirrors the quantum counterpart
    so that the same class name can be imported from the classical or quantum
    sub‑package.

    Parameters
    ----------
    kernels : Sequence[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
        List of kernel functions to fuse.  Each function must accept two
        tensors of shape (N, D) and return a tensor of shape (N,).
    weights : Sequence[float], optional
        Weight for each kernel.  If omitted, equal weighting is used.
    """

    def __init__(
        self,
        kernels: Sequence[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] | None = None,
        weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if kernels is None:
            kernels = [self.rbf_kernel, self.matern_kernel]
        self.kernel_funcs = list(kernels)
        if weights is None:
            weights = [1.0] * len(kernels)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    @staticmethod
    def rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
        """Standard RBF kernel."""
        diff = x - y
        return torch.exp(-gamma * torch.sum(diff * diff, dim=-1))

    @staticmethod
    def matern_kernel(
        x: torch.Tensor, y: torch.Tensor, nu: float = 1.5, lengthscale: float = 1.0
    ) -> torch.Tensor:
        """Matern kernel with ν = 1.5 or 2.5."""
        diff = torch.norm(x - y, dim=-1)
        if nu == 1.5:
            return (
                1.0 + torch.sqrt(3.0) * diff / lengthscale
            ) * torch.exp(-torch.sqrt(3.0) * diff / lengthscale)
        elif nu == 2.5:
            return (
                1.0
                + torch.sqrt(5.0) * diff / lengthscale
                + 5.0 * diff * diff / (3.0 * lengthscale * lengthscale)
            ) * torch.exp(-torch.sqrt(5.0) * diff / lengthscale)
        else:
            raise ValueError("Unsupported ν for Matern kernel.")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the fused kernel value for two batches of input vectors.

        Parameters
        ----------
        x, y : torch.Tensor
            Input tensors of shape (N, D).

        Returns
        -------
        torch.Tensor
            Kernel value of shape (N,).
        """
        values = torch.stack([func(x, y) for func in self.kernel_funcs], dim=-1)
        weighted = (self.weights * values).sum(dim=-1)
        return weighted

    def kernel_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """
        Compute the Gram matrix between two collections of feature vectors.

        Parameters
        ----------
        a, b : sequence of torch.Tensor
            Each element is a tensor of shape (D,).

        Returns
        -------
        np.ndarray
            Gram matrix of shape (len(a), len(b)).
        """
        return np.array(
            [
                [
                    self(
                        torch.tensor(x, dtype=torch.float32),
                        torch.tensor(y, dtype=torch.float32),
                    ).item()
                    for y in b
                ]
                for x in a
            ]
        )

    def set_weights(self, weights: Sequence[float]) -> None:
        """Update the kernel fusion weights."""
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def add_kernel(
        self,
        kernel_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        weight: float = 1.0,
    ) -> None:
        """Append a new kernel to the fusion."""
        self.kernel_funcs.append(kernel_func)
        self.weights = torch.cat(
            [self.weights, torch.tensor([weight], dtype=torch.float32)]
        )

__all__ = ["QuantumKernelMethod"]
