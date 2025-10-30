"""Classical RBF kernel module with trainable gamma and kernel matrix utilities.

Author: The OpenAI‑GPT‑LLM‑Engine
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence

class RBFKernel(nn.Module):
    """Trainable RBF kernel."""

    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class QuantumKernelMethod(nn.Module):
    """
    Wrapper that exposes a classical RBF kernel or a hybrid kernel.

    Parameters
    ----------
    mode : str, optional
        'rbf' for pure classical RBF kernel, 'hybrid' for weighted sum of
        classical and quantum kernels.  'quantum' is not supported here.
    gamma : float, optional
        Initial gamma for the RBF kernel.
    weight : float, optional
        Weight of the classical kernel in the hybrid mode.  The quantum
        contribution is `1-weight`.
    """

    def __init__(self, mode: str = "rbf", gamma: float = 1.0, weight: float = 0.5):
        super().__init__()
        if mode not in {"rbf", "hybrid"}:
            raise ValueError(f"Unsupported mode {mode!r}.")
        self.mode = mode
        self.rbf = RBFKernel(gamma)

        if mode == "hybrid":
            self.weight = weight

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        rbf_val = self.rbf(x, y)
        if self.mode == "rbf":
            return rbf_val.squeeze()
        # In hybrid mode we need a quantum contribution.  For the purpose of
        # this classical module we compute a dummy term that can be overridden
        # by the quantum implementation in the QML module.
        quantum_val = torch.zeros_like(rbf_val)
        return (self.weight * rbf_val + (1.0 - self.weight) * quantum_val).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  mode: str = "rbf", gamma: float = 1.0, weight: float = 0.5) -> np.ndarray:
    """
    Compute the Gram matrix between two sets of vectors.

    Parameters
    ----------
    a, b : sequences of 1‑D torch tensors
        The datasets for which the kernel matrix is required.
    mode : str
        Kernel mode passed to :class:`QuantumKernelMethod`.
    gamma : float
        Initial gamma for the RBF kernel.
    weight : float
        Weight for hybrid mode.
    """
    kernel = QuantumKernelMethod(mode=mode, gamma=gamma, weight=weight)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["QuantumKernelMethod", "kernel_matrix"]
