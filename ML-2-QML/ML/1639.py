"""Hybrid RBF kernel with GPU acceleration and batch‑wise distance computation."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import numpy as np

class QuantumKernelMethod(nn.Module):
    """Compute a classical RBF kernel between two tensors.

    Parameters
    ----------
    gamma : float, optional
        Width parameter of the RBF kernel.
    device : str or torch.device, optional
        Device on which to perform the computation (e.g., "cpu" or "cuda").

    The forward method accepts two torch tensors of shape (n, d) and (m, d)
    and returns the Gram matrix K_{ij} = exp(-gamma ||x_i - y_j||^2).
    """

    def __init__(self, gamma: float = 1.0, device: str | torch.device = 'cpu') -> None:
        super().__init__()
        self.gamma = gamma
        self.device = torch.device(device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are on the desired device
        x = x.to(self.device)
        y = y.to(self.device)
        # Broadcasted squared Euclidean distance
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (n, m, d)
        dist2 = torch.sum(diff * diff, dim=-1)  # (n, m)
        return torch.exp(-self.gamma * dist2)

    @staticmethod
    def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
        """Compute pairwise kernel matrix for sequences of tensors.

        Each element in ``a`` and ``b`` is expected to be a 1‑D tensor
        of the same dimensionality.  The function flattens them into 2‑D
        tensors, computes the kernel using :class:`QuantumKernelMethod`,
        and returns a NumPy array of shape (len(a), len(b)).
        """
        # Flatten and stack inputs
        a_flat = torch.cat([t.reshape(1, -1) if t.ndim == 1 else t for t in a], dim=0)
        b_flat = torch.cat([t.reshape(1, -1) if t.ndim == 1 else t for t in b], dim=0)
        kernel = QuantumKernelMethod(gamma=gamma, device=a_flat.device)
        return kernel(a_flat, b_flat).cpu().numpy()

__all__ = ["QuantumKernelMethod", "kernel_matrix"]
