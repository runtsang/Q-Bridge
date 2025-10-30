"""Hybrid quantum–classical kernel with a trainable phase shift."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn, optim


class ClassicalRBF(nn.Module):
    """Implements a simple RBF kernel with trainable gamma."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the k(x,y) = exp(-γ‖x‑y‖²)."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=1, keepdim=True))


class QuantumPhaseShift(nn.Module):
    """Quantum‑based phase shift that is linear in parameters and an angle‑tuned."""

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Randomly initialise phase parameters for each wire
        self.phases = nn.Parameter(torch.rand(n_wires, dtype=torch.float32) * 2 * np.pi)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute a simple interference term based on phase differences."""
        # Map the input vectors to phase angles
        theta_x = torch.matmul(x, self.phases)
        theta_y = torch.matmul(y, self.phases)
        # Interference term: cosine of phase difference
        return torch.cos(theta_x - theta_y).mean()


class HybridKernel(nn.Module):
    """Combines a classical RBF kernel with a quantum phase shift."""

    def __init__(self, gamma: float = 1.0, n_wires: int = 4) -> None:
        super().__init__()
        self.rbf = ClassicalRBF(gamma)
        self.qphase = QuantumPhaseShift(n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the product of classical RBF and quantum phase."""
        return self.rbf(x, y) * self.qphase(x, y)


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0, n_wires: int = 4) -> np.ndarray:
    kernel = HybridKernel(gamma, n_wires)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["ClassicalRBF", "QuantumPhaseShift", "HybridKernel", "kernel_matrix"]
