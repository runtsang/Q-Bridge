"""Enhanced classical RBF kernel with adaptive scaling and data normalization.

The :class:`QuantumKernelMethod` exposes a torch.nn.Module that computes an
RBF kernel with optional standardisation and a learnable width hyper‑parameter.
It also provides a convenient :func:`kernel_matrix` helper that works on
iterables of tensors.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler


class QuantumKernelMethod(nn.Module):
    """Classical RBF kernel with optional adaptive width and normalization.

    Parameters
    ----------
    gamma : float, default=1.0
        Initial Gaussian width.  If ``adaptive=True`` this value becomes a learnable
        parameter.
    normalize : bool, default=True
        Whether to standardise the data before kernel evaluation.
    adaptive : bool, default=False
        Whether ``gamma`` should be a trainable parameter.
    lr : float, default=1e-3
        Learning rate used when ``adaptive=True``.  It is stored for convenience
        but not used internally; optimisation is left to the caller.
    """

    def __init__(self, gamma: float = 1.0, *, normalize: bool = True,
                 adaptive: bool = False, lr: float = 1e-3) -> None:
        super().__init__()
        self.normalize = normalize
        self.adaptive = adaptive
        self.lr = lr
        if adaptive:
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        else:
            self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float32))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return x
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) + 1e-8
        return (x - mean) / std

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel between two 1‑D tensors."""
        x = self._normalize(x)
        y = self._normalize(y)
        diff = x - y
        sq_norm = torch.sum(diff * diff, dim=-1, keepdim=True)
        return torch.exp(-self.gamma * sq_norm).squeeze()

    @staticmethod
    def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                      *, gamma: float = 1.0, normalize: bool = True,
                      adaptive: bool = False) -> np.ndarray:
        """Convenience wrapper that returns a NumPy Gram matrix."""
        model = QuantumKernelMethod(gamma, normalize=normalize, adaptive=adaptive)
        # Fit the normalisation statistics on the combined data
        if normalize:
            all_data = torch.cat(a + b, dim=0)
            mean = all_data.mean(dim=0, keepdim=True)
            std = all_data.std(dim=0, keepdim=True) + 1e-8
            a = [(x - mean) / std for x in a]
            b = [(x - mean) / std for x in b]
        mat = torch.zeros((len(a), len(b)))
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                mat[i, j] = model.forward(x, y)
        return mat.cpu().numpy()


__all__ = ["QuantumKernelMethod"]
