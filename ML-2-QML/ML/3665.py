"""Hybrid classical kernel combining RBF with a learnable sampler network.

This module extends the baseline RBF kernel by augmenting it with a
probability‑based kernel derived from a lightweight neural sampler.
The weighted sum of the two kernels allows the end‑to‑end model to
learn data‑dependent feature maps while retaining the analytical
expressions of the RBF kernel.  The design mirrors the quantum
implementation in :mod:`QuantumKernelMethod__gen307_qml.py` and
facilitates direct comparison of classical and quantum scaling
behaviours.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SamplerQNN", "HybridKernel"]


class SamplerQNN(nn.Module):
    """A shallow neural sampler that maps 2‑dimensional inputs to a 4‑dim
    categorical probability distribution.  The output is softmaxed to
    guarantee a valid probability vector which is later used in a
    dot‑product kernel.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(8, 4, bias=True),
            nn.Softmax(dim=-1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)


class HybridKernel(nn.Module):
    """Weighted combination of an analytical RBF kernel and a sampler‑based
    kernel.  The ``alpha`` hyper‑parameter controls the relative
    contribution of each term.
    """

    def __init__(self, gamma: float = 1.0, alpha: float = 0.5) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.sampler = SamplerQNN()

    # -----------------------------------------------------------------
    # classical RBF kernel
    # -----------------------------------------------------------------
    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    # -----------------------------------------------------------------
    # sampler‑based kernel
    # -----------------------------------------------------------------
    def _sampler_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        px = self.sampler(x)
        py = self.sampler(y)
        return torch.sum(px * py, dim=-1, keepdim=True)

    # -----------------------------------------------------------------
    # combined kernel
    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return a scalar kernel value for two input vectors."""
        # reshape to 2‑D for broadcasting
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        k_rbf = self._rbf(x, y)
        k_samp = self._sampler_kernel(x, y)
        return self.alpha * k_rbf + (1.0 - self.alpha) * k_samp

    # -----------------------------------------------------------------
    # Gram matrix construction
    # -----------------------------------------------------------------
    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
        *,
        dtype: torch.dtype = torch.float32,
    ) -> np.ndarray:
        """Compute the Gram matrix between two collections of vectors."""
        K = np.zeros((len(a), len(b)), dtype=np.float64)
        for i, xi in enumerate(a):
            for j, yj in enumerate(b):
                K[i, j] = float(self.forward(xi.to(dtype), yj.to(dtype)).item())
        return K
