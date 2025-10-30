"""Hybrid kernel supporting multiple classical families and automatic gamma selection."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence, Optional, Literal


class HybridKernel(nn.Module):
    """Classical kernel module supporting RBF, polynomial, and linear kernels.

    Parameters
    ----------
    kernel_type: Literal['rbf', 'poly', 'linear'], default='rbf'
        The family of the kernel.
    gamma: float, optional
        Gaussian width for RBF and scaling coefficient for polynomial.
        If None and kernel_type=='rbf', gamma is set by the median trick.
    degree: int, default=3
        Degree of the polynomial kernel.
    coef0: float, default=1.0
        Independent term in the polynomial kernel.
    """

    def __init__(
        self,
        kernel_type: Literal["rbf", "poly", "linear"] = "rbf",
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 1.0,
    ) -> None:
        super().__init__()
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the radial‑basis kernel."""
        diff = x - y
        dist_sq = torch.sum(diff * diff, dim=-1, keepdim=True)
        return torch.exp(-self.gamma * dist_sq)

    def _poly(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the polynomial kernel."""
        dot = torch.sum(x * y, dim=-1, keepdim=True)
        return (self.gamma * dot + self.coef0) ** self.degree

    def _linear(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the linear kernel."""
        return torch.sum(x * y, dim=-1, keepdim=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return kernel value for two input vectors."""
        # Broadcast to (1, d) if needed
        x = x.view(1, -1)
        y = y.view(1, -1)

        if self.kernel_type == "rbf":
            if self.gamma is None:
                # Median trick: estimate gamma from pairwise distances of x and y
                diff = x - y
                dist_sq = torch.sum(diff * diff).item()
                self.gamma = 1.0 / (2.0 * dist_sq) if dist_sq > 0 else 1.0
            return self._rbf(x, y)
        elif self.kernel_type == "poly":
            return self._poly(x, y)
        elif self.kernel_type == "linear":
            return self._linear(x, y)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
        gamma: Optional[float] = None,
    ) -> np.ndarray:
        """Compute Gram matrix between two collections of vectors.

        Parameters
        ----------
        a, b : sequence of torch.Tensor
            Each element is a 1‑D tensor representing a data point.
        gamma : float, optional
            Override the instance gamma for this call.

        Returns
        -------
        np.ndarray
            Gram matrix of shape (len(a), len(b)).
        """
        old_gamma = self.gamma
        if gamma is not None:
            self.gamma = gamma

        matrix = torch.stack(
            [
                torch.stack([self.forward(x, y) for y in b])
                for x in a
            ],
            dim=0,
        )
        result = matrix.squeeze().cpu().numpy()

        # Restore old gamma
        self.gamma = old_gamma
        return result


__all__ = ["HybridKernel"]
