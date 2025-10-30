"""Hybrid kernel–attention model – classical implementation."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class KernelAttentionModel(nn.Module):
    """RBF kernel combined with a classical self‑attention block."""

    def __init__(self, embed_dim: int = 4, gamma: float = 1.0, n_qubits: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.gamma = gamma
        self.n_qubits = n_qubits  # kept for API compatibility

    # --------------------------------------------------------------------------- #
    # RBF kernel
    # --------------------------------------------------------------------------- #
    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def compute_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel value between two 1‑D tensors."""
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self._rbf(x, y).squeeze()

    # --------------------------------------------------------------------------- #
    # Classical self‑attention
    # --------------------------------------------------------------------------- #
    def self_attention(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Simple dot‑product attention implemented with torch."""
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

    # --------------------------------------------------------------------------- #
    # Matrix helpers
    # --------------------------------------------------------------------------- #
    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Gram matrix of the RBF kernel."""
        return np.array(
            [
                [self.compute_kernel(torch.tensor(x, dtype=torch.float32),
                                     torch.tensor(y, dtype=torch.float32))
                 for y in b]
                for x in a
            ]
        )

    def combined_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> np.ndarray:
        """Combine the kernel matrix with attention weights."""
        K = self.kernel_matrix(a, b)
        # Attention weights per row of ``a``; broadcast over columns
        attn_weights = np.array(
            [
                self.self_attention(rotation_params, entangle_params, x.reshape(1, -1))
               .squeeze()
                for x in a
            ]
        )
        return K * attn_weights[:, None]


__all__ = ["KernelAttentionModel"]
