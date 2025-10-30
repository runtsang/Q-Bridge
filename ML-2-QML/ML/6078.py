"""Hybrid kernel classifier – classical implementation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class KernalAnsatz(nn.Module):
    """Classical RBF kernel ansatz."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Wrapper for :class:`KernalAnsatz` that normalises inputs."""

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


class HybridKernelModel(nn.Module):
    """
    Classical kernel‑based classifier.

    Parameters
    ----------
    gamma : float
        RBF kernel width.  Larger values produce smoother decision boundaries.
    num_classes : int
        Number of output classes.
    support_vectors : torch.Tensor, optional
        Pre‑selected support vectors used to construct the kernel feature map.
        If ``None`` the model raises an error during ``forward``.
    """

    def __init__(self, gamma: float = 1.0, num_classes: int = 4, support_vectors: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.support_vectors = support_vectors
        self.linear = nn.Linear(1, num_classes)

    def _kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Internal RBF kernel computation."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using the pre‑computed support vectors."""
        if self.support_vectors is None:
            raise ValueError("Support vectors must be provided for inference.")
        # Compute kernel between input batch and support vectors
        k = self._kernel(x.unsqueeze(1), self.support_vectors.unsqueeze(0))
        logits = self.linear(k.squeeze(-1))
        return logits


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix", "HybridKernelModel"]
