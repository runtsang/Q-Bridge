"""Enhanced classical RBF kernel with learnable gamma and GPU support."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter

__all__ = ["QuantumKernelMethod", "kernel_matrix", "KernelFactory"]


class QuantumKernelMethod(nn.Module):
    """Learnable RBF kernel with optional GPU acceleration.

    Parameters
    ----------
    gamma : float or torch.Tensor, optional
        Initial value for the length‑scale parameter. If a float is
        supplied, it is wrapped in a ``torch.nn.Parameter`` and can be
        optimized during training.
    device : str or torch.device, optional
        Target device for tensor operations. Defaults to ``'cpu'``.
    """

    def __init__(self, gamma: float | torch.Tensor = 1.0, device: str | torch.device = "cpu"):
        super().__init__()
        if isinstance(gamma, float):
            gamma = torch.tensor(gamma, dtype=torch.float32, device=device)
        self.gamma = Parameter(gamma)
        self.device = torch.device(device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel value for two input vectors.

        The computation is performed on the same device as the instance.
        """
        x = x.to(self.device)
        y = y.to(self.device)
        diff = x - y
        sq_norm = torch.sum(diff * diff, dim=-1, keepdim=True)
        return torch.exp(-self.gamma * sq_norm)


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float | torch.Tensor = 1.0, device: str | torch.device = "cpu") -> np.ndarray:
    """
    Compute the Gram matrix between two collections of vectors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of 1‑D tensors.
    gamma : float or torch.Tensor, optional
        Length‑scale parameter for the RBF kernel.
    device : str or torch.device, optional
        Target device for intermediate tensors.
    """
    kernel = QuantumKernelMethod(gamma, device)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


def KernelFactory(kind: str, **kwargs) -> QuantumKernelMethod:
    """
    Simple factory that returns a concrete kernel implementation.

    Parameters
    ----------
    kind : str
        ``'classical'`` or ``'quantum'``. The quantum variant is a
        placeholder that simply forwards to :class:`QuantumKernelMethod`.
    kwargs : dict
        Forwarded to the kernel constructor.
    """
    if kind == "classical":
        return QuantumKernelMethod(**kwargs)
    elif kind == "quantum":
        # For the purpose of this demo, the quantum kernel is still
        # a classical RBF. Real quantum kernels will be provided in
        # the QML module.
        return QuantumKernelMethod(**kwargs)
    else:
        raise ValueError(f"Unknown kernel kind: {kind}")
