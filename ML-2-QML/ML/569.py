"""Enhanced classical RBF kernel with GPU support and shared interface."""

from __future__ import annotations

from typing import Sequence, List, Any

import numpy as np
import torch
from torch import nn

__all__ = ["KernelBase", "KernalKernelBase", "KernalAnsatz", "Kernel", "kernel_matrix"]

class KernelBase(nn.Module):
    """Abstract base enforcing a ``forward`` that returns a scalar kernel value."""
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class KernalKernelBase(KernelBase):
    """Base for the legacy ``KernalAnsatz`` and the new GPU‑RBF."""
    pass


class KernalAnsatz(nn.Module):
    """Legacy RBF kernel implementation (CPU only)."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(KernelBase):
    """GPU‑accelerated RBF kernel that can optionally learn ``gamma``."""
    def __init__(self, gamma: float = 1.0, learn_gamma: bool = False) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma)) if learn_gamma else torch.tensor(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b`` using the GPU‑RBF."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])
