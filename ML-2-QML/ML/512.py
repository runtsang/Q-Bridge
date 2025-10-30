"""Hybrid classical RBF kernel with learnable gamma and batch evaluation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter

class QuantumKernelMethod(nn.Module):
    """A classical RBF kernel with a trainable gamma.

    The kernel value is
        k(x, y) = exp(-γ‖x−y‖²).
    The gamma parameter can be frozen or optimised.  The implementation
    supports batched inputs and a convenient ``kernel_matrix`` helper.
    """

    def __init__(self, gamma: float = 1.0, requires_grad: bool = False) -> None:
        super().__init__()
        self.gamma = Parameter(torch.tensor(gamma, dtype=torch.float32), requires_grad=requires_grad)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return kernel values for two batches ``x`` and ``y``."""
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # shape (n, m, d)
        sq_dist = (diff * diff).sum(dim=-1)      # shape (n, m)
        return torch.exp(-self.gamma * sq_dist)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper returning the Gram matrix."""
        return self(a, b)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Return the Gram matrix for two collections of 1‑D tensors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Collections of samples.
    gamma : float, optional
        Default gamma used by the kernel.

    Returns
    -------
    np.ndarray
        Gram matrix of shape (len(a), len(b)).
    """
    kernel = QuantumKernelMethod(gamma)
    return kernel(torch.stack(a), torch.stack(b)).detach().cpu().numpy()

__all__ = ["QuantumKernelMethod", "kernel_matrix"]
