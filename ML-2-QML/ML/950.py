"""SharedKernel: classical RBF kernel with anisotropic support and multi‑gamma evaluation."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class SharedKernel(nn.Module):
    """Classical radial‑basis function kernel.

    Parameters
    ----------
    gamma : float or Sequence[float], optional
        Width of the Gaussian.  If a sequence is provided, an anisotropic
        kernel is used where each dimension has its own width.
    normalize : bool, optional
        If ``True`` the kernel matrix is divided by its maximum value.
    """
    def __init__(self, gamma: float | Sequence[float] = 1.0, normalize: bool = False) -> None:
        super().__init__()
        self.gamma = torch.tensor(gamma, dtype=torch.float32)
        self.normalize = normalize

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value ``k(x, y)``."""
        # Ensure inputs are 2‑D tensors
        x = x.unsqueeze(0) if x.ndim == 1 else x
        y = y.unsqueeze(0) if y.ndim == 1 else y
        diff = x - y
        if self.gamma.ndim == 0:
            sq_norm = torch.sum(diff * diff, dim=-1, keepdim=True)
            k = torch.exp(-self.gamma * sq_norm)
        else:
            # Anisotropic: element‑wise gamma
            sq_norm = torch.sum(diff * diff * self.gamma, dim=-1, keepdim=True)
            k = torch.exp(-sq_norm)
        return k.squeeze()

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return the Gram matrix ``K`` between two sets of samples."""
        a = a.unsqueeze(0) if a.ndim == 1 else a
        b = b.unsqueeze(0) if b.ndim == 1 else b
        diff = a[:, None, :] - b[None, :, :]
        if self.gamma.ndim == 0:
            sq_norm = torch.sum(diff * diff, dim=-1)
            K = torch.exp(-self.gamma * sq_norm)
        else:
            sq_norm = torch.sum(diff * diff * self.gamma, dim=-1)
            K = torch.exp(-sq_norm)
        if self.normalize:
            K = K / K.max()
        return K


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  gamma: float | Sequence[float] = 1.0, normalize: bool = False) -> np.ndarray:
    """Convenience wrapper that builds a :class:`SharedKernel` and returns a NumPy array."""
    kernel = SharedKernel(gamma, normalize)
    return kernel.kernel_matrix(torch.stack(a), torch.stack(b)).detach().cpu().numpy()


def kernel_matrix_multi(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                        gammas: Sequence[float | Sequence[float]]) -> dict[tuple, np.ndarray]:
    """Compute Gram matrices for a list of gamma configurations.

    Returns a dictionary mapping the gamma tuple to the corresponding matrix.
    """
    results = {}
    for g in gammas:
        results[tuple(g) if isinstance(g, Sequence) else (g,)] = kernel_matrix(a, b, g)
    return results


__all__ = ["SharedKernel", "kernel_matrix", "kernel_matrix_multi"]
