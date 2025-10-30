"""Hybrid kernel combining classical RBF with a neural network approximation.

This module defines a class `HybridKernelMethod` that extends PyTorch's nn.Module. It computes a kernel similarity between two
feature vectors using a Gaussian RBF kernel, then refines this similarity through a lightweight neural network.  The design
mirrors the original `QuantumKernelMethod` API, enabling seamless substitution in existing pipelines while adding a
learnable component that can adapt to domain‑specific data distributions.  The class also exposes a static `kernel_matrix`
utility that evaluates the Gram matrix for any iterable of tensors.

The implementation deliberately diverges from the seed by:
* Replacing the one‑parameter `KernalAnsatz` with a two‑stage RBF + NN pipeline.
* Introducing a small neural network that learns to map the raw RBF similarity to a more expressive similarity score.
* Providing a `forward` method that accepts batched tensors and returns a similarity scalar.

This hybrid approach allows classical preprocessing to be enriched with a data‑driven refinement step, which can be
further complemented by the quantum kernel defined in the QML partner module.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class HybridKernelMethod(nn.Module):
    """Hybrid kernel that combines an RBF similarity with a small neural network."""

    def __init__(self, gamma: float = 1.0, hidden_dim: int = 8) -> None:
        """
        Parameters
        ----------
        gamma : float
            RBF kernel width.
        hidden_dim : int
            Size of the hidden layer in the refinement network.
        """
        super().__init__()
        self.gamma = gamma
        # Small neural refiner: maps scalar similarity to refined similarity.
        self.refiner = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Keep output in [0, 1] like a kernel value.
        )

    def rbf_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the Gaussian RBF similarity."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Compute the hybrid kernel value for two input tensors.

        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of the same shape representing feature vectors.

        Returns
        -------
        torch.Tensor
            A scalar kernel value in [0, 1].
        """
        rbf = self.rbf_similarity(x, y)
        refined = self.refiner(rbf)
        return refined.squeeze()

    @staticmethod
    def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
        """
        Evaluate the Gram matrix between two collections of feature vectors.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Collections of 1‑D tensors.
        gamma : float
            RBF width used by the hybrid kernel.

        Returns
        -------
        np.ndarray
            2‑D array of kernel values.
        """
        kernel = HybridKernelMethod(gamma)
        return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["HybridKernelMethod"]
