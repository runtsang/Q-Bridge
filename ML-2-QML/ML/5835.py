"""Enhanced classical radial basis function kernel utilities with auto‑bandwidth and compositional support."""

from __future__ import annotations

from typing import Sequence, Optional, List, Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import pairwise_distances

class QuantumKernelMethod(nn.Module):
    """
    Classical kernel module that supports RBF kernels with optional auto‑bandwidth
    selection, multi‑output handling, and kernel composition.

    Parameters
    ----------
    gamma : float, optional
        Kernel width. Ignored when ``auto_bandwidth=True``.
    auto_bandwidth : bool, default=False
        If True, gamma is set to 1/(2 * median(pairwise_distance)**2).
    composition : List[Callable], optional
        List of kernel callables or string aliases to compose multiplicatively.
    """

    def __init__(
        self,
        gamma: Optional[float] = 1.0,
        auto_bandwidth: bool = False,
        composition: Optional[List[Callable]] = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.auto_bandwidth = auto_bandwidth
        self.composition = composition or []

    def _compute_gamma(self, x: torch.Tensor) -> float:
        """Compute median‑heuristic bandwidth."""
        xy = x.detach().cpu().numpy()
        dists = pairwise_distances(xy, metric="euclidean")
        median = np.median(dists)
        return 1.0 / (2.0 * median ** 2 + 1e-12)

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Standard RBF kernel."""
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        sq_dist = torch.sum(diff**2, dim=2)
        gamma = self.gamma
        if self.auto_bandwidth:
            gamma = self._compute_gamma(x)
        return torch.exp(-gamma * sq_dist)

    def _compose(self, k1: torch.Tensor, k2: torch.Tensor) -> torch.Tensor:
        """Multiplicative composition."""
        return k1 * k2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel matrix between ``x`` and ``y``.
        ``x`` and ``y`` must be 2‑D tensors of shape (n_samples, n_features).
        """
        k = self._rbf(x, y)
        for comp in self.composition:
            if callable(comp):
                k = self._compose(k, comp(x, y))
            else:
                # simple string alias for common kernels
                if comp == "linear":
                    k = self._compose(k, torch.mm(x, y.t()))
                elif comp == "poly":
                    k = self._compose(k, (torch.mm(x, y.t()) + 1) ** 2)
        return k

    @staticmethod
    def kernel_matrix(
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
        gamma: Optional[float] = 1.0,
        auto_bandwidth: bool = False,
    ) -> np.ndarray:
        """Convenience wrapper returning a NumPy array."""
        a_tensor = torch.stack(a)
        b_tensor = torch.stack(b)
        module = QuantumKernelMethod(gamma, auto_bandwidth)
        k = module(a_tensor, b_tensor)
        return k.detach().cpu().numpy()

__all__ = ["QuantumKernelMethod"]
