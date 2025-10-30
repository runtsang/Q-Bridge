"""Hybrid classical kernel with optional feature map.

This module defines :class:`HybridKernel` that implements a radial basis
function kernel with an optional neural network based feature extractor.
The implementation is a drop‑in replacement for the original
``QuantumKernelMethod`` class but adds a `feature_map` argument that can
take any :class:`torch.nn.Module`.  The public API mirrors the seed
module so existing scripts continue to run unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence, Optional

class HybridKernel(nn.Module):
    """Classical RBF kernel with an optional feature map.

    Parameters
    ----------
    gamma : float, default 1.0
        RBF bandwidth.
    feature_map : nn.Module, optional
        A neural network that maps raw inputs to a higher‑dimensional
        representation.  If ``None`` the raw vectors are used directly.
    """
    def __init__(self, gamma: float = 1.0, feature_map: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.feature_map = feature_map

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_map(x) if self.feature_map is not None else x

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the RBF kernel value between two 1‑D tensors."""
        x_t = self._transform(x).view(-1)
        y_t = self._transform(y).view(-1)
        diff = x_t - y_t
        return torch.exp(-self.gamma * torch.sum(diff * diff))

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  gamma: float = 1.0,
                  feature_map: Optional[nn.Module] = None) -> np.ndarray:
    """Compute Gram matrix using :class:`HybridKernel`."""
    kernel = HybridKernel(gamma=gamma, feature_map=feature_map)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["HybridKernel", "kernel_matrix"]
