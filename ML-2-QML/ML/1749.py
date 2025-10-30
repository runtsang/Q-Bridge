"""Hybrid kernel module with learnable gamma and GPU support."""

from __future__ import annotations

import torch
import numpy as np
from torch import nn
from typing import Sequence

class KernalAnsatz(nn.Module):
    """Classical RBF kernel ansatz with learnable gamma."""
    def __init__(self, gamma: float = 1.0, device: str | None = None) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Hybrid kernel module usable in scikit‑learn pipelines."""
    def __init__(self, gamma: float = 1.0, device: str | None = None) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma, device=device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two collections of tensors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
