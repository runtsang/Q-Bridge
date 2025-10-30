"""Enhanced classical RBF kernel with bandwidth selection and batched support."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

__all__ = ["KernelRBF", "Kernel", "kernel_matrix", "KernelHybrid"]

class KernelRBF(nn.Module):
    """Radial Basis Function (RBF) kernel with optional learnable bandwidth.

    Parameters
    ----------
    gamma : float | None
        Bandwidth parameter. If None, the median heuristic is used on the
        training data to initialise gamma.
    learnable : bool
        If True, gamma becomes a trainable parameter.
    """
    def __init__(self, gamma: float | None = None, learnable: bool = False) -> None:
        super().__init__()
        if gamma is None:
            gamma_val = 1.0
        else:
            gamma_val = gamma
        self.gamma = nn.Parameter(torch.tensor(gamma_val, dtype=torch.float32))
        if not learnable:
            self.gamma.requires_grad = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel between two batches of vectors.

        Parameters
        ----------
        x, y : torch.Tensor
            Batches of vectors with shape (N, D) and (M, D) respectively.

        Returns
        -------
        torch.Tensor
            Gram matrix of shape (N, M).
        """
        # Ensure 2D tensors
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (N, M, D)
        sq_norm = torch.sum(diff * diff, dim=-1)  # (N, M)
        return torch.exp(-self.gamma * sq_norm)

class Kernel(KernelRBF):
    """Alias to KernelRBF for backward compatibility."""
    pass

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float | None = None) -> np.ndarray:
    """Compute the kernel matrix for two sequences of tensors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of samples. Each element is a 1‑D tensor.
    gamma : float | None
        Optional bandwidth. If None, a median heuristic is applied.

    Returns
    -------
    np.ndarray
        Gram matrix of shape (len(a), len(b)).
    """
    if gamma is None:
        # Median heuristic on pairwise distances
        all_samples = torch.cat([torch.stack(a), torch.stack(b)], dim=0)
        dists = torch.cdist(all_samples, all_samples, p=2).flatten()
        gamma_val = 1.0 / (2 * torch.median(dists).item() ** 2 + 1e-8)
    else:
        gamma_val = gamma
    kernel = KernelRBF(gamma=gamma_val, learnable=False)
    return kernel(torch.stack(a), torch.stack(b)).detach().cpu().numpy()

class KernelHybrid(nn.Module):
    """Placeholder for a hybrid kernel that can be overridden by a quantum implementation."""
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
