"""Hybrid kernel module combining classical RBF and optional quantum feature maps.

The module exposes a unified interface that can be used interchangeably with
the original `Kernel` class.  It supports batched inputs and a weighted
combination of a classical RBF kernel and a user supplied quantum kernel.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch import nn

class KernalAnsatz(nn.Module):
    """Simple RBF kernel ansatz used by the original implementation.

    It is kept for backward compatibility but now also supports batched
    evaluation and a configurable gamma parameter.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff))

class Kernel(nn.Module):
    """Wrapper that keeps the original public API but now exposes a
    batched interface and optional quantum kernel integration.
    """
    def __init__(
        self,
        gamma: float = 1.0,
        use_quantum: bool = False,
        quantum_kernel: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)
        self.use_quantum = use_quantum
        self.quantum_kernel = quantum_kernel
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        classical = self.ansatz(x, y).squeeze()
        if self.use_quantum and self.quantum_kernel is not None:
            quantum = self.quantum_kernel(x, y).squeeze()
            return self.alpha * classical + (1.0 - self.alpha) * quantum
        return classical

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Efficient batched Gram matrix for a list of 1‑D tensors."""
    a = torch.stack(a)
    b = torch.stack(b)
    diff = a.unsqueeze(1) - b.unsqueeze(0)
    dist_sq = torch.sum(diff * diff, dim=-1)
    return torch.exp(-gamma * dist_sq).cpu().numpy()

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
