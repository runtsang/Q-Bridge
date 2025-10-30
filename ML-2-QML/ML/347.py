"""Hybrid kernel module with classical RBF and optional quantum placeholder.

The module implements a classical radial basis function (RBF) kernel
with automatic gamma selection via a simple heuristic.  It also
provides a thin wrapper for a quantum kernel that is only available
in the QML version, ensuring API compatibility between the two
implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

class Kernel(nn.Module):
    """Protocol for kernel callables."""
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

@dataclass
class KernelConfig:
    """Hyper‑parameters for kernel construction."""
    gamma: float | None = None  # None triggers automatic estimation

class KernalRBF(nn.Module):
    """Classical RBF kernel with optional gamma auto‑selection."""
    def __init__(self, gamma: float | None = None) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        sq_norm = torch.sum(diff * diff, dim=-1, keepdim=True)
        return torch.exp(-self.gamma * sq_norm)

    def fit(self, X: torch.Tensor) -> None:
        if self.gamma is None:
            dists = torch.cdist(X, X, p=2) ** 2
            self.gamma = 1.0 / dists.mean().item()

class QuantumKernelPlaceholder:
    """Placeholder for quantum kernel – raises error when used."""
    def __init__(self, *_, **__):
        raise NotImplementedError("Quantum kernel is only available in the QML module.")

class QuantumKernelMethod(nn.Module):
    """Hybrid kernel that can operate in classical or quantum mode.

    Parameters
    ----------
    mode : str, default 'rbf'
        Either 'rbf' for the classical radial basis function kernel
        or 'quantum' for the quantum kernel (not implemented here).
    config : KernelConfig | None, default None
        Hyper‑parameter container.  For 'rbf' mode, 'gamma' can be
        None to trigger an automatic estimate.
    """
    def __init__(self, mode: str = "rbf", config: KernelConfig | None = None) -> None:
        super().__init__()
        self.mode = mode
        self.config = config or KernelConfig()
        if mode == "rbf":
            self.kernel = KernalRBF(gamma=self.config.gamma)
        elif mode == "quantum":
            self.kernel = QuantumKernelPlaceholder()
        else:
            raise ValueError(f"Unsupported mode {mode!r}")

    def fit(self, X: torch.Tensor) -> None:
        """Fit kernel hyper‑parameters."""
        if self.mode == "rbf":
            self.kernel.fit(X)
        # quantum mode has no fit step in this module

    def kernel_matrix(self, X: Sequence[torch.Tensor], Y: Sequence[torch.Tensor] | None = None) -> np.ndarray:
        """Return Gram matrix between X and Y."""
        if Y is None:
            Y = X
        return np.array([[self.kernel(x, y).item() for y in Y] for x in X])

__all__ = ["QuantumKernelMethod", "KernelConfig", "KernalRBF"]
