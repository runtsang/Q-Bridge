"""Hybrid trainable RBF kernel with optional neural‑network feature mapping.

This module extends the original `Kernel` class by adding:
- a learnable `gamma` parameter that can be optimized with back‑propagation;
- an optional user‑supplied neural network (`mapping`) that transforms
  the input data before the kernel is applied.
- a `kernel_matrix` method that operates on sequences of tensors and
  returns a NumPy array, keeping the original API compatible.
"""

from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
import torch
from torch import nn


class QuantumKernelMethod(nn.Module):
    """
    Classical RBF kernel with an optional trainable width and feature mapping.

    Parameters
    ----------
    gamma : float
        Width of the RBF kernel. If ``trainable=True``, this value becomes a
        learnable parameter.
    trainable : bool, default=False
        Whether ``gamma`` should be optimized during training.
    mapping : nn.Module | None, default=None
        Optional neural network that maps the input data to a new feature
        space before the kernel is evaluated.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        trainable: bool = False,
        mapping: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if trainable:
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        else:
            self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))
        self.mapping = mapping

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel between two batches of samples.

        Parameters
        ----------
        x, y : torch.Tensor
            Batches of shape (batch, features). The tensors are flattened
            to 2‑D if necessary.

        Returns
        -------
        torch.Tensor
            Kernel values of shape (batch,).
        """
        if self.mapping is not None:
            x = self.mapping(x)
            y = self.mapping(y)
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        diff = x - y
        sq_dist = torch.sum(diff * diff, dim=-1, keepdim=True)
        return torch.exp(-self.gamma * sq_dist).squeeze()

    def kernel_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """
        Compute the Gram matrix for two collections of samples.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Sequences of tensors that will be stacked into 2‑D arrays.

        Returns
        -------
        np.ndarray
            Gram matrix of shape (len(a), len(b)).
        """
        A = torch.stack([x.view(-1, x.shape[-1]) for x in a])
        B = torch.stack([x.view(-1, x.shape[-1]) for x in b])
        if self.mapping is not None:
            A = self.mapping(A)
            B = self.mapping(B)
        diff = A[:, None, :] - B[None, :, :]
        sq_dist = torch.sum(diff * diff, dim=-1)
        K = torch.exp(-self.gamma * sq_dist)
        return K.detach().cpu().numpy()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gamma={self.gamma.item():.4f}, trainable={self.gamma.requires_grad})"


__all__ = ["QuantumKernelMethod"]
