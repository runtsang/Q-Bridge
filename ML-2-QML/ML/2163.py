"""Enhanced classical RBF kernel module with batch support and normalisation."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class QuantumKernelMethod(nn.Module):
    """Classical radial basis function kernel with optional normalisation.

    Parameters
    ----------
    gamma : float, default 1.0
        Width parameter of the RBF kernel.
    normalize : bool, default False
        If ``True`` the Gram matrix is normalised by the diagonal elements
        so that ``K[i,i] == 1``.
    """
    def __init__(self, gamma: float = 1.0, normalize: bool = False) -> None:
        super().__init__()
        self.gamma = gamma
        self.normalize = normalize

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return ``K(x, y)`` for each pair of rows in ``x`` and ``y``."""
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        diff = x[:, None, :] - y[None, :, :]
        return torch.exp(-self.gamma * (diff**2).sum(-1))

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute the Gram matrix between two sequences of feature vectors."""
        a = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in a])
        b = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in b])
        K = self.forward(a, b).cpu().numpy()
        if self.normalize:
            Kaa = self.forward(a, a).cpu().numpy()
            Kbb = self.forward(b, b).cpu().numpy()
            diag_a = np.diag(Kaa)
            diag_b = np.diag(Kbb)
            denom = np.sqrt(np.outer(diag_a, diag_b))
            K = K / denom
        return K


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  gamma: float = 1.0, normalize: bool = False) -> np.ndarray:
    """Convenience wrapper that returns the kernel matrix for two datasets."""
    qkm = QuantumKernelMethod(gamma=gamma, normalize=normalize)
    return qkm.kernel_matrix(a, b)


__all__ = ["QuantumKernelMethod", "kernel_matrix"]
