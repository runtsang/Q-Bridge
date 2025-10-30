"""Enhanced classical RBF kernel with learnable bandwidth and batch support."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class QuantumKernelMethod(nn.Module):
    """A classical RBF kernel with optional learnable bandwidth.

    The kernel can be evaluated on batches of data and can optionally
    learn the bandwidth parameter ``gamma`` via gradient descent.
    """

    def __init__(self, gamma: float = 1.0, learnable: bool = False) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        if not learnable:
            self.gamma.requires_grad = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel matrix between two batches.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_x, features) or (features,)
        y : torch.Tensor
            Shape (batch_y, features) or (features,)

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (batch_x, batch_y)
        """
        # Ensure 2D shape
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if y.ndim == 1:
            y = y.unsqueeze(0)

        # Broadcast difference
        diff = x[:, None, :] - y[None, :, :]
        dist_sq = torch.sum(diff * diff, dim=-1)
        return torch.exp(-self.gamma * dist_sq)

    @staticmethod
    def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
        """
        Compute the Gram matrix between two collections of feature vectors.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Each element should be a 1‑D tensor of shape (features,).
        gamma : float, optional
            Kernel bandwidth.

        Returns
        -------
        np.ndarray
            Gram matrix of shape (len(a), len(b)).
        """
        x = torch.stack(a)
        y = torch.stack(b)
        kernel = QuantumKernelMethod(gamma)
        with torch.no_grad():
            return kernel(x, y).cpu().numpy()


__all__ = ["QuantumKernelMethod"]
