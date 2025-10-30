"""Pure‑classical implementation of the EstimatorQNN hybrid model.

The module exposes two interchangeable backends:
- A vanilla feed‑forward regressor (seed from EstimatorQNN.py).
- A kernel‑based regressor that uses a classical radial‑basis function (RBF) kernel.

Both share the same constructor signature; the `use_kernel` flag selects the backend.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

import torch
from torch import nn


class RBFKernel(nn.Module):
    """Classical RBF kernel compatible with the quantum interface."""

    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel matrix between `x` and `y`.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, D)
        y : torch.Tensor
            Shape (A, D)

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (B, A)
        """
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (B, A, D)
        dist_sq = (diff**2).sum(dim=-1)         # (B, A)
        return torch.exp(-self.gamma * dist_sq)


class EstimatorQNN(nn.Module):
    """Hybrid regressor with classical feed‑forward and optional kernel backend."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Sequence[int] = (8, 4),
        output_dim: int = 1,
        use_kernel: bool = False,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.use_kernel = use_kernel
        if self.use_kernel:
            self.kernel = RBFKernel(gamma)
            # Linear read‑out after kernel
            self.readout = nn.Linear(1, output_dim)
        else:
            layers = []
            prev_dim = input_dim
            for h in hidden_dims:
                layers.extend([nn.Linear(prev_dim, h), nn.Tanh()])
                prev_dim = h
            layers.append(nn.Linear(prev_dim, output_dim))
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, x_ref: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (B, D).
        x_ref : torch.Tensor, optional
            Reference batch used for kernel evaluation when `use_kernel=True`.
            If omitted, a self‑kernel is computed (useful for toy examples).

        Returns
        -------
        torch.Tensor
            Predicted scalar per sample.
        """
        if self.use_kernel:
            if x_ref is None:
                x_ref = x
            k = self.kernel(x, x_ref)  # (B, A)
            # For each sample, take mean similarity to all references
            feats = k.mean(dim=-1, keepdim=True)  # (B, 1)
            return self.readout(feats)
        else:
            return self.net(x)

__all__ = ["EstimatorQNN"]
