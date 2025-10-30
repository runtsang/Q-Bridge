"""Hybrid classical kernel with an optional sampler network.

The class implements a radial basis function (RBF) kernel and, if requested,
amplifies the similarity by the inner product of a small neural sampler.
This mirrors the structure of the original `QuantumKernelMethod` while
adding a learnable component.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

__all__ = ["HybridKernelMethod", "SamplerQNN"]


class SamplerQNN(nn.Module):
    """A tiny 2‑to‑2 softmax sampler that can be attached to the kernel.

    It provides a learnable weighting of the input vectors before the RBF
    similarity is computed.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)


class HybridKernelMethod(nn.Module):
    """Classical RBF kernel optionally enhanced with a sampler network.

    Parameters
    ----------
    gamma : float, default=1.0
        Width of the RBF kernel.
    use_sampler : bool, default=False
        If ``True`` a :class:`~SamplerQNN` is applied to each input before
        the kernel is computed.
    """
    def __init__(self, gamma: float = 1.0, use_sampler: bool = False) -> None:
        super().__init__()
        self.gamma = gamma
        self.use_sampler = use_sampler
        if use_sampler:
            self.sampler = SamplerQNN()

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the raw RBF similarity."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the (possibly sampler‑weighted) kernel value."""
        k = self._rbf(x, y)
        if self.use_sampler:
            sx = self.sampler(x)
            sy = self.sampler(y)
            k *= torch.sum(sx * sy, dim=-1, keepdim=True)
        return k.squeeze()

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return the full Gram matrix between two sets of samples."""
        return np.array(
            [[self.forward(x.unsqueeze(0), y.unsqueeze(0)).item() for y in b] for x in a]
        )
