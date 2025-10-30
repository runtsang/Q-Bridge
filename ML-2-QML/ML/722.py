"""Enhanced RBF kernel with learnable hyperparameters and batched support."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class Kernel(nn.Module):
    """Learnable Gaussian‑process style kernel.

    The kernel has two trainable parameters:

    * ``lengthscale`` – controls the width of the RBF.
    * ``variance`` – scales the overall magnitude.

    The forward method accepts two tensors ``x`` and ``y`` of shape
    ``(batch, features)`` and returns the Gram matrix
    ``K_{ij} = variance * exp(-||x_i - y_j||^2 / (2 * lengthscale^2))``.
    The implementation is fully vectorised and works on GPU.
    """

    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0) -> None:
        super().__init__()
        self.lengthscale = nn.Parameter(torch.tensor(lengthscale, dtype=torch.float32))
        self.variance = nn.Parameter(torch.tensor(variance, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel matrix between ``x`` and ``y``.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(batch_x, features)``.
        y : torch.Tensor
            Tensor of shape ``(batch_y, features)``.

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape ``(batch_x, batch_y)``.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)

        x_norm = (x**2).sum(dim=1, keepdim=True)
        y_norm = (y**2).sum(dim=1, keepdim=True)
        sq_dist = x_norm + y_norm.t() - 2.0 * x @ y.t()

        scaled = -sq_dist / (2.0 * self.lengthscale**2)
        return self.variance * torch.exp(scaled)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute the Gram matrix between two sequences of 1‑D tensors.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Sequences of tensors of shape ``(features,)`` or ``(batch, features)``.

        Returns
        -------
        np.ndarray
            NumPy array of shape ``(len(a), len(b))``.
        """
        a_tensor = torch.stack([t.to(self.lengthscale.device) for t in a])
        b_tensor = torch.stack([t.to(self.lengthscale.device) for t in b])
        with torch.no_grad():
            return self.forward(a_tensor, b_tensor).cpu().numpy()


__all__ = ["Kernel"]
