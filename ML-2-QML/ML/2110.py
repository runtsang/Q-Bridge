"""Hybrid RBF‑quantum kernel module – classical side."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class Embedder(nn.Module):
    """Learnable linear embedding that normalizes data before RBF."""
    def __init__(self, in_dim: int, out_dim: int = 32):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.linear(x, self.weight, self.bias)


class KernalAnsatz(nn.Module):
    """Classical RBF kernel with optional embedding."""
    def __init__(self, in_dim: int, gamma: float = 1.0, embed: bool = False) -> None:
        super().__init__()
        self.gamma = gamma
        self.embed = embed
        self.embedder = Embedder(in_dim) if embed else None

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if self.embed and self.embedder is not None:
            x = self.embedder(x)
            y = self.embedder(y)
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """RBF kernel module that wraps :class:`KernalAnsatz`."""
    def __init__(self, in_dim: int, gamma: float = 1.0, embed: bool = False) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(in_dim, gamma, embed)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: Sequence[Tensor], b: Sequence[Tensor], in_dim: int, gamma: float = 1.0, embed: bool = False) -> np.ndarray:
    kernel = Kernel(in_dim, gamma, embed)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["Embedder", "KernalAnsatz", "Kernel", "kernel_matrix"]
