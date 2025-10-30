"""Hybrid classical‑quantum kernel module with RBF and a classical surrogate for a quantum feature map."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

class HybridKernel(nn.Module):
    """Hybrid kernel that multiplies a Gaussian RBF with a classical surrogate
    of a quantum feature map.

    The surrogate is a random orthonormal projection of dimensionality
    2**depth.  The depth parameter controls the size of the feature space
    and therefore the expressive power of the quantum component.  The
    class supports batched inputs and can be used as a drop‑in replacement
    for the original :class:`KernalAnsatz` in the seed project.
    """

    def __init__(self, gamma: float = 1.0, depth: int = 2, seed: int | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.depth = depth
        self._proj: torch.Tensor | None = None
        self._seed = seed

    def _ensure_proj(self, dim: int) -> None:
        """Create a random orthonormal projection matrix if it does not exist."""
        if self._proj is None:
            rng = np.random.default_rng(self._seed)
            A = rng.standard_normal((2 ** self.depth, dim))
            q, _ = np.linalg.qr(A)
            self._proj = torch.tensor(q, dtype=torch.float32, device="cpu")

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def _quantum_surrogate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Classical surrogate for a quantum feature map."""
        self._ensure_proj(x.shape[-1])
        fx = torch.matmul(x, self._proj.t())
        fy = torch.matmul(y, self._proj.t())
        return torch.exp(-torch.sum((fx - fy) ** 2, dim=-1, keepdim=True))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the hybrid kernel value for two batches of vectors."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        rbf = self._rbf(x, y)
        q_surrogate = self._quantum_surrogate(x, y)
        return rbf * q_surrogate

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0, depth: int = 2) -> np.ndarray:
    """Compute the Gram matrix between two collections of tensors."""
    kernel = HybridKernel(gamma=gamma, depth=depth)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["HybridKernel", "kernel_matrix"]
