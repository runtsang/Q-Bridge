"""Hybrid kernel class with classical self‑attention weighting.

This module implements a hybrid RBF kernel that first computes
self‑attention weights using a lightweight classical transformer
block.  The attention weighted samples are then fed into a
standard radial‑basis‑function kernel.  The design mirrors the
`QuantumKernelMethod` interface so that training scripts can
switch between the classical and quantum variants without any
modification.

Typical usage::

    from QuantumKernelMethod import QuantumKernelMethod
    model = QuantumKernelMethod(embed_dim=4, gamma=0.5)
    K = model(x, y)   # returns kernel matrix of shape (len(x), len(y))

"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
#  Classical attention helper (mirrors the Qiskit implementation)
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention:
    """Simple multi‑head attention with a single head.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input vectors (must match the feature size).
    """

    def __init__(self, embed_dim: int) -> None:
        self.embed_dim = embed_dim

        # For reproducibility we pre‑generate random linear layers
        self.W_q = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_k = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute attention‑weighted representation.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (embed_dim, embed_dim), used as query matrix.
        entangle_params : np.ndarray
            Shape (embed_dim, embed_dim), used as key matrix.
        inputs : np.ndarray
            Shape (n_samples, embed_dim).

        Returns
        -------
        np.ndarray
            Weighted values, shape (n_samples, embed_dim).
        """
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)

        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        weighted = scores @ value
        return weighted.numpy()


# --------------------------------------------------------------------------- #
#  Classical RBF kernel
# --------------------------------------------------------------------------- #
class KernalAnsatz(nn.Module):
    """Purely classical RBF kernel computation."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Wrapper around :class:`KernalAnsatz` that normalizes input shapes."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


# --------------------------------------------------------------------------- #
#  Hybrid kernel class
# --------------------------------------------------------------------------- #
class QuantumKernelMethod(nn.Module):
    """
    Hybrid kernel that applies self‑attention weighting before
    evaluating an RBF kernel.  The class name is intentionally
    the same as the quantum counterpart so that a single import
    can be used in both classical and quantum pipelines.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of input features; must match attention weights.
    gamma : float, optional
        RBF kernel scaling parameter.
    """

    def __init__(self, embed_dim: int, gamma: float = 1.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.gamma = gamma
        self.attention = ClassicalSelfAttention(embed_dim)
        self.kernel = Kernel(gamma)

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the attention‑weighted kernel matrix.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the query matrix.
        entangle_params : np.ndarray
            Parameters for the key matrix.
        x : np.ndarray
            Shape (n_samples_x, embed_dim).
        y : np.ndarray
            Shape (n_samples_y, embed_dim).

        Returns
        -------
        np.ndarray
            Kernel matrix of shape (n_samples_x, n_samples_y).
        """
        # 1. Apply classical attention to each batch
        x_att = self.attention.forward(rotation_params, entangle_params, x)
        y_att = self.attention.forward(rotation_params, entangle_params, y)

        # 2. Compute kernel between weighted representations
        k_mat = np.array(
            [
                [self.kernel(torch.as_tensor(xi), torch.as_tensor(yi)).item()]
                for xi in x_att
                for yi in y_att
            ],
        ).reshape(len(x_att), len(y_att))
        return k_mat


def kernel_matrix(
    a: Sequence[np.ndarray],
    b: Sequence[np.ndarray],
    embed_dim: int,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Convenience wrapper that constructs a :class:`QuantumKernelMethod`
    instance and evaluates the Gram matrix between two collections of
    samples.

    Parameters
    ----------
    a, b : Sequence[np.ndarray]
        Each element should be a feature vector of shape (embed_dim,).
    embed_dim : int
        Dimensionality of the input vectors.
    gamma : float, optional
        RBF kernel scaling parameter.

    Returns
    -------
    np.ndarray
        Gram matrix of shape (len(a), len(b)).
    """
    method = QuantumKernelMethod(embed_dim, gamma)
    rotation = np.random.randn(4 * embed_dim).reshape(4, embed_dim)
    entangle = np.random.randn(4 * embed_dim).reshape(4, embed_dim)
    return method(rotation, entangle, np.array(a), np.array(b))


__all__ = ["QuantumKernelMethod", "kernel_matrix"]
