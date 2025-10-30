"""Hybrid classical kernel with trainable RBF and optional quantum weighting.

This module extends the original simple RBF kernel by making the
γ‑parameter learnable and by providing an optional quantum kernel
that can be blended in a differentiable way.  The implementation
remains fully classical so that it can be used in standard
PyTorch pipelines.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence, Optional, Callable

class KernalAnsatz(nn.Module):
    """Trainable radial‑basis‑function ansatz.

    The γ hyper‑parameter becomes a learnable `nn.Parameter`,
    allowing the model to adapt its bandwidth during training.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x and y are expected to be 1‑D or batch‑flattened
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Hybrid kernel that can optionally mix a quantum kernel.

    Parameters
    ----------
    gamma : float
        Initial value for the RBF bandwidth.
    quantum_kernel : Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
        A callable that returns a quantum kernel value for a pair of
        feature vectors.  If ``None`` the kernel reverts to pure RBF.
    weight : float
        Weight for the quantum kernel in the blending operation
        (only used when ``quantum_kernel`` is provided).
    """
    def __init__(
        self,
        gamma: float = 1.0,
        quantum_kernel: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)
        self.quantum_kernel = quantum_kernel
        self.weight = weight

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        rbf = self.ansatz(x, y).squeeze()

        if self.quantum_kernel is None:
            return rbf

        q = self.quantum_kernel(x.squeeze(), y.squeeze())
        # Blend classically and quantumly
        return (1.0 - self.weight) * rbf + self.weight * q

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix for two collections of feature vectors.

    The function creates a ``Kernel`` instance with the provided
    ``gamma`` and returns an ``np.ndarray`` containing the pairwise
    kernel values.  No quantum kernel is used in this helper.
    """
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
