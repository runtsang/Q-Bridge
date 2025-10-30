"""Hybrid classical kernel framework combining RBF with optional sampler network.

The HybridKernel class allows optional preprocessing of data via a
neural sampler network before applying the radial basis function kernel.
It supports batch inputs and can be used with PyTorch tensors or NumPy arrays.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable, Sequence

class HybridKernel(nn.Module):
    """
    Hybrid classical kernel that optionally applies a sampler network
    to the data before computing an RBF kernel.

    Parameters
    ----------
    gamma : float, optional
        RBF kernel width parameter.
    sampler : Callable[[torch.Tensor], torch.Tensor] | None, optional
        A callable that maps input data to a new representation.
        If None, identity is applied.
    """

    def __init__(self, gamma: float = 1.0, sampler: Callable[[torch.Tensor], torch.Tensor] | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.sampler = sampler if sampler is not None else lambda x: x

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel value between x and y.

        Parameters
        ----------
        x, y : torch.Tensor
            1-D tensors representing two data points.

        Returns
        -------
        torch.Tensor
            Kernel value as a scalar tensor.
        """
        # Preprocess with sampler
        x_p = self.sampler(x)
        y_p = self.sampler(y)
        diff = x_p - y_p
        return torch.exp(-self.gamma * torch.sum(diff * diff))

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix between sequences a and b."""
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

# Auxiliary Sampler network based on the provided SamplerQNN
class SamplerNN(nn.Module):
    """
    Simple neural network sampler that maps 2D inputs to 2D outputs.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

# Example usage:
# sampler = SamplerNN()
# kernel = HybridKernel(gamma=0.5, sampler=sampler.forward)
# k_val = kernel(torch.tensor([1.0, 2.0]), torch.tensor([1.5, 1.8]))
