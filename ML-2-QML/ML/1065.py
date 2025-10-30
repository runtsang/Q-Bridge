"""Vectorized, GPU‑friendly RBF kernel for classical ML."""

from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
import torch
from torch import nn


class QuantumKernel(nn.Module):
    """
    Classical RBF kernel with optional GPU acceleration and caching.

    Parameters
    ----------
    gamma : float, default=1.0
        Kernel width.
    use_gpu : bool, default=False
        If True, tensors are moved to the default CUDA device.
    normalize : bool, default=False
        If True, normalise the kernel matrix to unit diagonal.
    """

    def __init__(self, gamma: float = 1.0, use_gpu: bool = False, normalize: bool = False) -> None:
        super().__init__()
        self.gamma = gamma
        self.use_gpu = use_gpu
        self.normalize = normalize
        self._cache: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel value between two tensors.

        Parameters
        ----------
        x, y : torch.Tensor
            Input vectors of shape ``(d,)`` or ``(n, d)``.
        """
        if self.use_gpu:
            x, y = x.cuda(), y.cuda()
        diff = x.unsqueeze(-2) - y.unsqueeze(-3)
        sq_norm = torch.sum(diff * diff, dim=-1)
        return torch.exp(-self.gamma * sq_norm)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Vectorised Gram matrix computation with optional caching.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Sequences of input vectors.
        """
        if self._cache is not None:
            return self._cache.cpu().numpy()
        a_tensor = torch.stack(a).float()
        b_tensor = torch.stack(b).float()
        gram = self.forward(a_tensor, b_tensor)
        if self.normalize:
            diag = torch.diag(gram)
            gram = gram / torch.sqrt(diag.unsqueeze(-1) * diag.unsqueeze(0))
        self._cache = gram
        return gram.cpu().numpy()


__all__ = ["QuantumKernel"]
