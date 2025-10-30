"""Hybrid classical kernel module with gradient‑enabled RBF and optional quantum kernel integration.

This module keeps the original API while adding a trainable gamma and the ability to
incorporate a quantum kernel supplied by a callback.  It can be used with PyTorch
optimizers to jointly learn the RBF hyper‑parameter and the quantum kernel parameters.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Callable, Sequence, Optional

class QuantumKernelMethod__gen110(nn.Module):
    """Hybrid kernel combining a differentiable RBF kernel and an optional quantum kernel.

    Parameters
    ----------
    gamma : float, optional
        Initial value for the RBF kernel width.  It is wrapped in a ``nn.Parameter`` so
        that it can be optimized by a PyTorch optimizer.
    quantum_kernel : Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional
        Callback that returns a kernel matrix for two batches of feature vectors.  The
        callback is expected to return a tensor of shape ``(m, n)`` where ``m`` and ``n``
        are the batch sizes of the inputs.  It is typically supplied by the QML module.
    """
    def __init__(self, gamma: float = 1.0,
                 quantum_kernel: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        self.quantum_kernel = quantum_kernel

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the combined kernel matrix for two batches of feature vectors.

        The classical RBF part is computed as ``exp(-gamma * ||x - y||^2)``.  If a
        quantum kernel callback is provided, the result is a weighted sum of the
        classical and quantum kernels.
        """
        # Classical RBF
        x = x.unsqueeze(1)  # (m, 1, d)
        y = y.unsqueeze(0)  # (1, n, d)
        diff = x - y
        rbf = torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1))
        if self.quantum_kernel is None:
            return rbf
        q_mat = self.quantum_kernel(x.squeeze(1), y.squeeze(0))
        # Simple equal weighting; the weights could be learned if desired
        return 0.5 * rbf + 0.5 * q_mat

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix using the classical RBF kernel.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of feature tensors.  Each tensor is expected to be one‑dimensional.
    gamma : float, optional
        Kernel width hyper‑parameter.
    """
    model = QuantumKernelMethod__gen110(gamma)
    return np.array([[model(x, y).item() for y in b] for x in a])

def hybrid_kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                         gamma: float = 1.0,
                         quantum_kernel: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None) -> np.ndarray:
    """Compute Gram matrix using the HybridKernel.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of feature tensors.
    gamma : float, optional
        RBF kernel width.
    quantum_kernel : Callable, optional
        Quantum kernel callback.
    """
    model = QuantumKernelMethod__gen110(gamma, quantum_kernel)
    return np.array([[model(x, y).item() for y in b] for x in a])

__all__ = ["QuantumKernelMethod__gen110", "kernel_matrix", "hybrid_kernel_matrix"]
