"""Hybrid classical kernel and sampler implementation.

This module extends the legacy `QuantumKernelMethod.py` by providing
a trainable RBF kernel, a lightweight neural sampler, and utilities for
computing Gram matrices.  The API is kept compatible with the original
implementation while offering additional flexibility for downstream
experiments.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "SamplerQNN",
    "QuantumKernelMethod",
]


class KernalAnsatz(nn.Module):
    """Trainable RBF kernel ansatz.

    Parameters
    ----------
    gamma : float, optional
        Initial width hyper‑parameter.  It becomes a learnable
        parameter when ``requires_grad=True``.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        # Shape: (batch, 1)
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Convenience wrapper that exposes a single ``KernalAnsatz`` instance."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two collections of vectors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


class SamplerQNN(nn.Module):
    """A lightweight neural sampler that maps 2‑dimensional inputs to a
    probability distribution over two classes.  The network is fully
    differentiable and can be trained jointly with the kernel.
    """

    def __init__(self, hidden_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


class QuantumKernelMethod:
    """
    Hybrid kernel object that exposes both a classical RBF kernel and a
    neural sampler.  The public API is intentionally similar to the legacy
    implementation so downstream code can switch between the classical
    and quantum back‑ends with minimal changes.

    Parameters
    ----------
    gamma : float, optional
        Width of the RBF kernel.  If ``None`` a default value of 1.0 is used.
    hidden_dim : int, optional
        Hidden layer size for the SamplerQNN.
    """

    def __init__(
        self,
        gamma: float | None = 1.0,
        hidden_dim: int = 4,
    ) -> None:
        self.kernel = Kernel(gamma if gamma is not None else 1.0)
        self.sampler = SamplerQNN(hidden_dim)

    def transform(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the neural sampler to the data."""
        return self.sampler(inputs)

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """Return the RBF Gram matrix between two lists of tensors."""
        return kernel_matrix(a, b, gamma=self.kernel.ansatz.gamma.item())

    def combined_kernel(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """Compute a kernel matrix after the data have been mapped by the
        sampler.  Useful when the sampler is trained jointly with the
        downstream model.
        """
        a_mapped = [self.transform(x) for x in a]
        b_mapped = [self.transform(y) for y in b]
        return self.kernel_matrix(a_mapped, b_mapped)
