"""Enhanced classical RBF kernel with learnable gamma and optional batch support."""

from __future__ import annotations

import torch
from torch import nn
import numpy as np
from typing import Sequence, Tuple

class KernalAnsatz(nn.Module):
    """Learnable RBF kernel implementation."""
    def __init__(self, gamma: float = 1.0, trainable: bool = False) -> None:
        super().__init__()
        if trainable:
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        else:
            self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))
        self.trainable = trainable

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return exp(-γ||x−y||²)."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """RBF kernel module that wraps :class:`KernalAnsatz`."""
    def __init__(self, gamma: float = 1.0, trainable: bool = False) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma, trainable=trainable)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0,
                  trainable: bool = False) -> np.ndarray:
    """Compute Gram matrix between datasets a and b."""
    kernel = Kernel(gamma, trainable=trainable)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
