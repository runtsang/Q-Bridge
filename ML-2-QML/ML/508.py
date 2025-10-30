"""Enhanced classical RBF kernel with learnable gamma and batch support."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence, Union, Optional

class QuantumKernelEnhanced(nn.Module):
    """
    Classical Radial Basis Function (RBF) kernel with optional learnable
    gamma and efficient batch processing.
    """
    def __init__(self, gamma: float = 1.0, learnable: bool = False) -> None:
        super().__init__()
        if learnable:
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        else:
            self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel between two batches of samples.
        """
        x = x.float()
        y = y.float()
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (n, m, d)
        sq_dist = torch.sum(diff * diff, dim=-1)  # (n, m)
        gamma = self.gamma if isinstance(self.gamma, float) else self.gamma.squeeze()
        return torch.exp(-gamma * sq_dist)

def kernel_matrix(a: Sequence[Union[torch.Tensor, np.ndarray]],
                  b: Sequence[Union[torch.Tensor, np.ndarray]],
                  gamma: float = 1.0,
                  learnable: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute the Gram matrix between two collections of samples.
    """
    if not isinstance(a[0], torch.Tensor):
        a = [torch.from_numpy(np.asarray(x)) for x in a]
    if not isinstance(b[0], torch.Tensor):
        b = [torch.from_numpy(np.asarray(x)) for x in b]
    a = torch.stack(a)
    b = torch.stack(b)
    kernel = QuantumKernelEnhanced(gamma=gamma, learnable=learnable)
    K = kernel(a, b)
    return K.numpy() if not learnable else K

# Backwards‑compatibility aliases
KernalAnsatz = QuantumKernelEnhanced
Kernel = QuantumKernelEnhanced

__all__ = ["QuantumKernelEnhanced", "kernel_matrix", "KernalAnsatz", "Kernel"]
