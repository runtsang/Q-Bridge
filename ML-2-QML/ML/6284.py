"""Hybrid kernel with classical RBF and self‑attention.

This module exposes a single ``QuantumKernelAttention`` class that
behaves like the original ``Kernel`` from QuantumKernelMethod.py but
augments it with a classical self‑attention layer.  The class is fully
PyTorch‑based, so it can be used in any neural‑network training loop
without requiring a quantum back‑end.

The public API mirrors the original ``kernel_matrix`` helper so that
existing code continues to run unchanged, while the new ``attention_type``
argument allows the user to switch between a plain RBF kernel and a
kernel that first projects the inputs through a self‑attention mechanism.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn


class ClassicalKernel(nn.Module):
    """RBF kernel implemented with PyTorch tensors."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute exp(-γ‖x−y‖²) for 1‑D tensors."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class ClassicalSelfAttention:
    """Simple dot‑product self‑attention that mimics the quantum interface."""

    def __init__(self, embed_dim: int = 4) -> None:
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Apply a linear projection followed by soft‑max weighting."""
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


class QuantumKernelAttention(nn.Module):
    """
    Hybrid kernel that optionally pre‑processes the data with a
    classical self‑attention block before computing an RBF kernel.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        attention_type: str = "none",
        embed_dim: int = 4,
    ) -> None:
        super().__init__()
        self.kernel = ClassicalKernel(gamma)
        self.attention_type = attention_type
        if attention_type == "classical":
            self.attention = ClassicalSelfAttention(embed_dim)
        else:
            self.attention = None

    def forward(
        self,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        rotation_params: np.ndarray | None = None,
        entangle_params: np.ndarray | None = None,
    ) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)

        if self.attention is not None:
            if rotation_params is None or entangle_params is None:
                raise ValueError("Attention parameters must be provided.")
            # Project inputs through the attention layer
            x = torch.tensor(
                self.attention.run(rotation_params, entangle_params, x.numpy()),
                dtype=torch.float32,
            )
            y = torch.tensor(
                self.attention.run(rotation_params, entangle_params, y.numpy()),
                dtype=torch.float32,
            )

        return self.kernel(x, y)

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor | np.ndarray],
        b: Sequence[torch.Tensor | np.ndarray],
        rotation_params: np.ndarray | None = None,
        entangle_params: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the Gram matrix between two datasets."""
        return np.array(
            [
                [
                    self.forward(x, y, rotation_params, entangle_params).item()
                    for y in b
                ]
                for x in a
            ]
        )


def kernel_matrix(
    a: Sequence[torch.Tensor | np.ndarray],
    b: Sequence[torch.Tensor | np.ndarray],
    gamma: float = 1.0,
    attention_type: str = "none",
    rotation_params: np.ndarray | None = None,
    entangle_params: np.ndarray | None = None,
) -> np.ndarray:
    """Convenience wrapper that mirrors the original API."""
    qa = QuantumKernelAttention(
        gamma=gamma,
        attention_type=attention_type,
    )
    return qa.kernel_matrix(a, b, rotation_params, entangle_params)


__all__ = ["QuantumKernelAttention", "kernel_matrix"]
