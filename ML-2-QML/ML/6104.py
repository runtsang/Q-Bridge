"""Enhanced classical RBF kernel with trainable gamma and utility helpers.

The original seed provided a fixed RBF kernel. This extension turns the
kernel width into a learnable torch.nn.Parameter, enabling gradient‑based
optimization. The API remains identical: `forward(x, y)` returns a
scalar kernel value and a static `kernel_matrix(a, b)` computes the
Gram matrix.
"""

import numpy as np
import torch
from torch import nn
from typing import Sequence

class QuantumKernelMethod(nn.Module):
    """Trainable RBF kernel implementation.

    Parameters
    ----------
    gamma : float | torch.Tensor, optional
        Initial kernel width. If a tensor is provided it is registered
        as a learnable parameter.
    """
    def __init__(self, gamma: float | torch.Tensor = 1.0) -> None:
        super().__init__()
        if isinstance(gamma, torch.Tensor):
            self.gamma = nn.Parameter(gamma)
        else:
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel between two tensors."""
        # Ensure 2D shapes
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # shape (len(x), len(y), features)
        sq_dist = torch.sum(diff * diff, dim=-1)  # shape (len(x), len(y))
        return torch.exp(-self.gamma * sq_dist)

    @staticmethod
    def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                      gamma: float | torch.Tensor | None = None) -> np.ndarray:
        """Compute Gram matrix between sequences of tensors.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Collections of input vectors.
        gamma : float | torch.Tensor | None, optional
            If provided, overrides the kernel width used by the kernel.
        """
        if gamma is None:
            kernel = QuantumKernelMethod()
        else:
            kernel = QuantumKernelMethod(gamma)
        return np.array([[kernel.forward(x, y).item() for y in b] for x in a])

__all__ = ["QuantumKernelMethod"]
