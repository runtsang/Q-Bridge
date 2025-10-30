"""Hybrid classical‑quantum kernel module with trainable fusion."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

# Original classical kernel classes
class KernalAnsatz(nn.Module):
    """RBF kernel ansatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper around KernalAnsatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix for a set of vectors using the classical RBF kernel."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class QuantumKernelMethod(nn.Module):
    """Hybrid kernel that fuses a classical RBF kernel with a quantum kernel."""
    def __init__(self, gamma: float = 1.0, alpha: float = 0.5) -> None:
        super().__init__()
        self.classical = Kernel(gamma)
        self.alpha = nn.Parameter(torch.tensor(alpha))
    def forward(self, x: torch.Tensor, y: torch.Tensor, quantum: torch.Tensor) -> torch.Tensor:
        """Return the fused kernel value."""
        classical_val = self.classical(x, y)
        return self.alpha * quantum + (1.0 - self.alpha) * classical_val
    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], quantum_mat: np.ndarray) -> np.ndarray:
        """Return a fused Gram matrix given a pre‑computed quantum Gram matrix."""
        classical_mat = kernel_matrix(a, b, self.classical.ansatz.gamma)
        return self.alpha.item() * quantum_mat + (1.0 - self.alpha.item()) * classical_mat

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix", "QuantumKernelMethod"]
