"""Hybrid kernel combining classical RBF mapping and a fully connected layer.

The module exposes a PyTorch implementation of the kernel and an
auxiliary `kernel_matrix` routine that can be used for SVM or kernel
regression experiments.  The design intentionally mirrors the
quantum counterpart so that the same public API can be swapped
between the two back‑ends.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

class RBFAnsatz(nn.Module):
    """Radial‑basis‑function mapping used inside :class:`HybridKernel`."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        # compute exp(-gamma * ||x-y||^2)
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class HybridKernel(nn.Module):
    """Classical hybrid kernel that first applies an RBF mapping and
    then passes the result through a learnable fully‑connected layer.
    """
    def __init__(self, gamma: float = 1.0, n_features: int = 1) -> None:
        super().__init__()
        self.rbf = RBFAnsatz(gamma)
        # The fully‑connected layer operates on the scalar kernel value
        self.fc = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # RBF similarity
        k = self.rbf(x, y).squeeze()
        # feed through fully connected layer
        # reshape to match Linear input dimension
        k = k.view(-1, 1)
        out = torch.tanh(self.fc(k)).mean(dim=0)
        return out

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  gamma: float = 1.0) -> np.ndarray:
    """Return the Gram matrix between two collections of feature vectors
    using :class:`HybridKernel`.
    """
    kernel = HybridKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["RBFAnsatz", "HybridKernel", "kernel_matrix"]
