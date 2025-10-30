"""Hybrid classical estimator combining RBF kernel and feed‑forward network.

The model first maps inputs into a higher‑dimensional feature space using a
classical RBF kernel against a set of support vectors.  The resulting
feature vector is fed into a small neural network that outputs a single
regression value.

This design keeps the entire pipeline purely classical while retaining
the expressive power of kernel methods and neural networks.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Sequence

class RBFKernel(nn.Module):
    """Radial basis function kernel with learnable gamma."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return exp(-gamma * ||x - y||^2)."""
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # shape (batch, n_support, dim)
        dist_sq = torch.sum(diff ** 2, dim=-1)
        return torch.exp(-self.gamma * dist_sq)

class HybridEstimator(nn.Module):
    """
    Classical hybrid estimator.

    Parameters
    ----------
    support_vectors : torch.Tensor
        Reference points used in the RBF kernel.  Shape (n_support, dim).
    hidden_dims : Sequence[int]
        Sizes of hidden layers in the feed‑forward network.
    gamma : float, optional
        Initial value for the RBF kernel width.
    """
    def __init__(
        self,
        support_vectors: torch.Tensor,
        hidden_dims: Sequence[int] = (64, 32),
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.support_vectors = nn.Parameter(
            support_vectors.clone().detach(), requires_grad=False
        )
        self.kernel = RBFKernel(gamma)
        layers = []
        input_dim = len(support_vectors)
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.Tanh())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (batch, dim).

        Returns
        -------
        torch.Tensor
            Regression output of shape (batch, 1).
        """
        # Compute kernel matrix between input and support vectors
        k = self.kernel(x, self.support_vectors)  # (batch, n_support)
        return self.net(k)

__all__ = ["HybridEstimator"]
