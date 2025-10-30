"""Hybrid self-attention module combining classical attention with an RBF kernel for similarity scoring."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

class HybridSelfAttention:
    """
    Classical self-attention that uses a radial basis function kernel to compute
    similarity between query and key representations.  The kernel can be
    swapped for a quantum kernel implementation without changing the
    interface.
    """

    def __init__(self, embed_dim: int, gamma: float = 1.0) -> None:
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of the input vectors.
        gamma : float, optional
            RBF kernel bandwidth.  Defaults to 1.0.
        """
        self.embed_dim = embed_dim
        self.gamma = gamma

    def _rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute the pairwise RBF kernel matrix between X and Y.

        Parameters
        ----------
        X : torch.Tensor
            Query matrix of shape (batch, embed_dim).
        Y : torch.Tensor
            Key matrix of shape (batch, embed_dim).

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (batch, batch).
        """
        diff = X.unsqueeze(1) - Y.unsqueeze(0)  # (batch, batch, embed_dim)
        dist_sq = (diff ** 2).sum(dim=-1)
        return torch.exp(-self.gamma * dist_sq)

    def run(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> np.ndarray:
        """
        Execute self-attention on the input batch.

        Parameters
        ----------
        inputs : np.ndarray
            Input data of shape (batch, embed_dim).
        rotation_params : np.ndarray
            Parameters for the rotation matrix, shape (embed_dim, embed_dim).
        entangle_params : np.ndarray
            Parameters for the entanglement matrix, shape (embed_dim, embed_dim).

        Returns
        -------
        np.ndarray
            Output of shape (batch, embed_dim).
        """
        inputs = torch.as_tensor(inputs, dtype=torch.float32)
        rotation = torch.as_tensor(rotation_params, dtype=torch.float32).reshape(
            self.embed_dim, -1
        )
        entangle = torch.as_tensor(entangle_params, dtype=torch.float32).reshape(
            self.embed_dim, -1
        )

        # Classical query/key/value projections
        query = inputs @ rotation
        key = inputs @ entangle
        value = inputs

        # Compute attention scores via RBF kernel
        kernel = self._rbf_kernel(query, key)
        scores = F.softmax(kernel / np.sqrt(self.embed_dim), dim=-1)

        # Weighted sum of values
        output = scores @ value
        return output.numpy()
