"""Hybrid classical kernel with neural network feature mapping."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class QuantumKernel(nn.Module):
    """Classical kernel that learns a feature map via an MLP and applies an RBF kernel.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input data.
    hidden_dims : Sequence[int], optional
        Sizes of hidden layers in the feature extractor.
    gamma : float | None, optional
        RBF kernel bandwidth. If ``None`` the kernel defaults to a linear kernel.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] | None = None,
        gamma: float | None = 1.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma

        # Build feature extractor
        layers: list[nn.Module] = []
        prev_dim = input_dim
        if hidden_dims:
            for h in hidden_dims:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(nn.ReLU())
                prev_dim = h
        layers.append(nn.Linear(prev_dim, prev_dim))  # keep same dim as last hidden
        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value k(x, y)."""
        # Map to feature space
        fx = self.feature_extractor(x)
        fy = self.feature_extractor(y)

        diff = fx - fy
        if self.gamma is None:
            # Linear kernel
            return torch.sum(fx * fy, dim=-1, keepdim=True)
        sq_norm = torch.sum(diff * diff, dim=-1, keepdim=True)
        return torch.exp(-self.gamma * sq_norm)


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float | None = 1.0) -> np.ndarray:
    """Compute the Gram matrix for two batches of tensors."""
    extractor = QuantumKernel(input_dim=a[0].shape[-1], hidden_dims=None, gamma=gamma)
    return np.array([[extractor(x, y).item() for y in b] for x in a])


__all__ = ["QuantumKernel", "kernel_matrix"]
