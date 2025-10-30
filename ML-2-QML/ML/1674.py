"""Classical Gaussian RBF kernel with GPU support and batched interface."""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable

class KernelMethod(nn.Module):
    """
    Classical RBF kernel.

    Parameters
    ----------
    gamma : float, optional
        Width parameter.
    device : torch.device | str, optional
        Device for computations.
    dtype : torch.dtype, optional
        Tensor dtype.

    Notes
    -----
    Adds GPU acceleration, vectorised Gram matrix, and flexible
    forward that accepts single or batched tensors.
    """

    def __init__(self,
                 gamma: float = 1.0,
                 device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.gamma = gamma
        self.device = torch.device(device)
        self.dtype = dtype

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel between batches of vectors.

        Parameters
        ----------
        x, y : torch.Tensor
            2‑D tensors of shape (batch, dim).

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (len(x), len(y)).
        """
        x = x.to(self.device, dtype=self.dtype)
        y = y.to(self.device, dtype=self.dtype)
        diff = x[:, None, :] - y[None, :, :]
        dist2 = (diff ** 2).sum(-1)
        return torch.exp(-self.gamma * dist2)

    def kernel_matrix(self,
                      a: Iterable[torch.Tensor],
                      b: Iterable[torch.Tensor]) -> torch.Tensor:
        """
        Evaluate the Gram matrix between two collections of vectors.

        Parameters
        ----------
        a, b : iterable of torch.Tensor
            Each yields a 1‑D tensor of shape (dim,).

        Returns
        -------
        torch.Tensor
            Matrix of shape (len(a), len(b)).
        """
        a_batch = torch.stack([t for t in a], dim=0)
        b_batch = torch.stack([t for t in b], dim=0)
        return self.forward(a_batch, b_batch)

__all__ = ["KernelMethod"]
