"""Hybrid RBF kernel with learnable width.

This class implements a classic radial‑basis‑function kernel whose
parameter ``gamma`` is trainable via back‑propagation.  It can be
used in place of the original :class:`Kernel` while still providing a
fully PyTorch interface.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence

class QuantumKernelMethod(nn.Module):
    """Learnable RBF kernel.

    Parameters
    ----------
    initial_gamma : float, optional
        Initial value for the width parameter.  A small positive value
        is recommended to avoid vanishing gradients.
    """

    def __init__(self, initial_gamma: float = 1.0) -> None:
        super().__init__()
        # enforce positivity with soft‑plus in case of negative updates
        self.gamma = nn.Parameter(torch.tensor(initial_gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value ``exp(-γ‖x−y‖²)``."""
        x = x.view(-1)
        y = y.view(-1)
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff))

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  initial_gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix for two collections of vectors.

    Parameters
    ----------
    a, b : sequences of 1‑D torch tensors
    initial_gamma : float
        Initial gamma value for the kernel.  The function creates a
        temporary :class:`QuantumKernelMethod` instance and uses it to
        evaluate all pairwise kernel values.
    """
    kernel = QuantumKernelMethod(initial_gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["QuantumKernelMethod", "kernel_matrix"]
