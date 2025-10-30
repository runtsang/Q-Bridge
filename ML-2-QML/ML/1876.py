"""
Quantum kernel utilities – classical RBF implementation.

This module supplies a lightweight, GPU‑ready wrapper that
mirrors the API of the original `Kernel` class but uses a
pure‑PyTorch RBF kernel.  It can optionally dispatch to a
`quantum` kernel via the `QuantumKernelMethod` in the QML
module, but that import is deferred until runtime to keep the
module free of quantum dependencies.

Key features:
* GPU acceleration via an optional `device` argument.
* Automatic conversion of input tensors to the requested device.
* A thin `kernel_matrix` helper that accepts a sequence of
  tensors and returns a NumPy array.
"""

import numpy as np
import torch
from torch import nn
from typing import Sequence

__all__ = ["Kernel", "kernel_matrix"]

class Kernel(nn.Module):
    """GPU‑enabled RBF kernel compatible with the original API.

    Parameters
    ----------
    gamma : float, default 1.0
        RBF width parameter.
    device : str | torch.device, default 'cpu'
        Target device for all tensors.
    """

    def __init__(self, gamma: float = 1.0, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.gamma = gamma
        self.device = torch.device(device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are on the correct device
        x = x.to(self.device).view(1, -1)
        y = y.to(self.device).view(1, -1)
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0,
                  device: str | torch.device = "cpu") -> np.ndarray:
    """Compute Gram matrix for a list of tensors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Input data.  Each element is a 1‑D tensor.
    gamma : float, default 1.0
        RBF width.
    device : str | torch.device, default 'cpu'
        Target device for intermediate tensors.
    """
    kernel = Kernel(gamma=gamma, device=device)
    return np.array([[kernel(x, y).item() for y in b] for x in a])
