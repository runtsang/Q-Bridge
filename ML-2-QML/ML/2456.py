"""
Classical self‑attention that incorporates a learnable RBF kernel.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SelfAttentionHybrid(nn.Module):
    """
    Classical self‑attention module that uses a radial‑basis‑function kernel to
    weight the attention scores.  The kernel parameters are fixed but the
    attention weights are still learnable through the rotation and entangle
    matrices supplied at runtime.
    """

    def __init__(self, embed_dim: int, gamma: float = 1.0):
        """
        Parameters
        ----------
        embed_dim : int
            Dimension of the embedding space.
        gamma : float, default=1.0
            RBF kernel width.  Larger values make the kernel sharper.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.gamma = gamma

    @staticmethod
    def _rbf_kernel(a: torch.Tensor, b: torch.Tensor, gamma: float) -> torch.Tensor:
        """
        Compute the RBF kernel matrix between two sets of vectors.

        Parameters
        ----------
        a : Tensor of shape (N, D)
        b : Tensor of shape (M, D)
        gamma : float
            Kernel width.

        Returns
        -------
        Tensor of shape (N, M)
        """
        diff = a.unsqueeze(1) - b.unsqueeze(0)  # (N, M, D)
        dist_sq = (diff ** 2).sum(dim=-1)  # (N, M)
        return torch.exp(-gamma * dist_sq)

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Run a self‑attention step.

        Parameters
        ----------
        rotation_params : ndarray of shape (3 * embed_dim,)
            Rotation angles for each qubit (used only to mimic the quantum
            interface).
        entangle_params : ndarray of shape (embed_dim - 1,)
            Entangling angles (also only for interface compatibility).
        inputs : ndarray of shape (N, embed_dim)
            Input embeddings.

        Returns
        -------
        ndarray of shape (N, embed_dim)
            The attended representations.
        """
        # Convert to torch tensors for efficient linear algebra
        inp = torch.as_tensor(inputs, dtype=torch.float32)

        # Build query, key, value matrices
        q = inp @ torch.as_tensor(
            rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        k = inp @ torch.as_tensor(
            entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        v = inp

        # Compute RBF kernel similarity between queries and keys
        scores = self._rbf_kernel(q, k, self.gamma)

        # Soft‑max normalisation
        scores = torch.softmax(scores, dim=-1)

        # Weighted sum over values
        out = scores @ v
        return out.numpy()


__all__ = ["SelfAttentionHybrid"]
