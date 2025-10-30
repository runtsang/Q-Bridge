"""Hybrid kernel module with classical RBF and optional quantum feature mapping.

The new :class:`HybridKernel` exposes a dual‑mode interface:
* ``mode='rbf'`` – vanilla RBF kernel.
* ``mode='qrbf'`` – hybrid kernel that uses a trainable quantum circuit as an
  feature‑encoding map.  The circuit is differentiable with torch back‑end,
  so the kernel can be forward‑propagated during training.

The original :class:`Kernel` and :func:`kernel_matrix` are kept for backward
compatibility.
"""

from __future__ import annotations

from typing import Sequence, Callable

import numpy as np
import torch
from torch import nn


class KernalAnsatz(nn.Module):
    """Utility class that re‑implements the seed RBF kernel.

    Parameters
    ----------
    gamma : float, optional
        RBF bandwidth parameter.  The default value 1.0 matches the seed
        implementation.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Classic RBF kernel module that wraps :class:`KernalAnsatz`."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


class HybridKernel(nn.Module):
    """Hybrid kernel that can operate in classical RBF mode or quantum‑augmented mode.

    Parameters
    ----------
    gamma : float, optional
        RBF bandwidth for the classical mode.
    mode : {'rbf', 'qrbf'}, default 'rbf'
        Kernel mode.  ``'rbf'`` uses the classical RBF.  ``'qrbf'`` uses a
        user‑supplied quantum ansatz that must be a callable returning a
        scalar torch tensor.
    quantum_ansatz : Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional
        Call‑able that implements the quantum kernel.  Required if ``mode='qrbf'``.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        mode: str = "rbf",
        quantum_ansatz: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.mode = mode
        if mode == "rbf":
            self.ansatz = KernalAnsatz(gamma)
        elif mode == "qrbf":
            if quantum_ansatz is None:
                raise ValueError("``quantum_ansatz`` must be provided for mode='qrbf'")
            self.ansatz = quantum_ansatz
        else:
            raise ValueError(f"Unsupported mode: {mode!r}")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the kernel value between ``x`` and ``y``."""
        return self.ansatz(x, y)


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b``."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["KernalAnsatz", "Kernel", "HybridKernel", "kernel_matrix"]
