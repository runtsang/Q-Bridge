"""Hybrid kernel method combining classical RBF kernel and fully‑connected layer.

This module extends the original QuantumKernelMethod by adding a unified
class :class:`HybridKernelMethod` that can compute classical RBF kernels,
classical fully‑connected layer expectations, and expose legacy
interfaces for backward compatibility.  The design follows a
*combination* scaling paradigm: classical operations are lightweight
and quantum routines are encapsulated in optional sub‑modules that can
be enabled when a quantum backend is available.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Legacy wrappers – keep the original API for users that still import
# KernalAnsatz or Kernel from this module.
# --------------------------------------------------------------------------- #

class KernalAnsatz(nn.Module):
    """Legacy wrapper around the classical RBF kernel.

    The class is kept for backward compatibility.  It delegates the
    computation to :class:`HybridKernelMethod`.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return HybridKernelMethod.classical_kernel(x, y, self.gamma)


class Kernel(nn.Module):
    """Legacy wrapper that exposes a single ``forward`` method.

    The wrapper simply forwards to :class:`HybridKernelMethod` and
    squeezes the result to match the original API.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two batches using the classical RBF kernel."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# --------------------------------------------------------------------------- #
# Classic fully‑connected layer – mimics the quantum FCL example.
# --------------------------------------------------------------------------- #

def FCL() -> nn.Module:
    """Return a classical fully‑connected layer that mimics the quantum example."""

    class FullyConnectedLayer(nn.Module):
        def __init__(self, n_features: int = 1) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().numpy()

    return FullyConnectedLayer()


# --------------------------------------------------------------------------- #
# Unified hybrid API.
# --------------------------------------------------------------------------- #

class HybridKernelMethod:
    """Unified API for classical and quantum kernels and fully‑connected layers.

    Parameters
    ----------
    gamma : float, optional
        Width parameter for the RBF kernel.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma

    @staticmethod
    def classical_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
        """Compute the classical RBF kernel between two vectors."""
        diff = x - y
        return torch.exp(-gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    @staticmethod
    def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float) -> np.ndarray:
        """Return the Gram matrix using the classical RBF kernel."""
        return np.array([[HybridKernelMethod.classical_kernel(x, y, gamma).item() for y in b]
                         for x in a])

    @staticmethod
    def fully_connected_layer(thetas: Iterable[float], n_features: int = 1) -> np.ndarray:
        """Compute the expectation of a classical fully‑connected layer."""
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        linear = nn.Linear(n_features, 1)
        expectation = torch.tanh(linear(values)).mean(dim=0)
        return expectation.detach().numpy()
