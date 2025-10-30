"""Hybrid radial basis function kernel with a learnable exponent.

This module builds on the original classical RBF kernel by making the
exponent a trainable parameter (or a small neural network).  The
interface matches the original ``Kernel`` class so that existing
pipelines can be dropped in with minimal changes.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch
from torch import nn


class KernalAnsatz(nn.Module):
    """Base class for kernel ansatzes."""
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RBFAnsatz(KernalAnsatz):
    """RBF kernel with a learnable gamma parameter."""
    def __init__(self, gamma: Union[float, nn.Parameter] = 1.0):
        super().__init__()
        if isinstance(gamma, float):
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        else:
            self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        dist_sq = torch.sum(diff ** 2, dim=-1, keepdim=True)
        return torch.exp(-self.gamma * dist_sq)


class Kernel(nn.Module):
    """Wrapper that evaluates the kernel for two samples."""
    def __init__(self, gamma: Union[float, nn.Parameter] = 1.0):
        super().__init__()
        self.ansatz = RBFAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are 1‑D tensors of shape (1, features)
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  gamma: Union[float, nn.Parameter] = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two collections of samples."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
