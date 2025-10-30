"""Hybrid self‑attention implemented purely in PyTorch.

The class mirrors the original SelfAttention interface but replaces the
dot‑product similarity with an RBF kernel that emulates a quantum kernel.
It is fully classical and can be used as a drop‑in replacement.
"""

import numpy as np
import torch
from torch import nn
from typing import Tuple

class HybridSelfAttention(nn.Module):
    """
    Classical self‑attention that uses an RBF kernel for the attention
    scores, thereby approximating a quantum‑kernel similarity.
    """
    def __init__(self, embed_dim: int, gamma: float = 1.0):
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of the input embeddings.
        gamma : float, optional
            RBF kernel width.  Larger values produce a sharper similarity
            profile that mimics a more expressive quantum feature map.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.gamma = gamma
        # Linear projections for query, key and value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel matrix K(x, y) = exp(-gamma ||x - y||^2).
        """
        # x: (batch, seq_len, dim)
        # y: (batch, seq_len, dim)
        x_exp = x.unsqueeze(2)  # (batch, seq_len, 1, dim)
        y_exp = y.unsqueeze(1)  # (batch, 1, seq_len, dim)
        diff = x_exp - y_exp   # (batch, seq_len, seq_len, dim)
        dist2 = (diff ** 2).sum(-1)  # (batch, seq_len, seq_len)
        return torch.exp(-self.gamma * dist2)

    def forward(self,
                inputs: torch.Tensor,
                rotation_params: torch.Tensor,
                entangle_params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid attention.

        Parameters
        ----------
        inputs : torch.Tensor
            Input embeddings of shape (batch, seq_len, embed_dim).
        rotation_params : torch.Tensor
            Parameters used to rotate the input before projection.
        entangle_params : torch.Tensor
            Parameters used to entangle the input before projection.

        Returns
        -------
        torch.Tensor
            Output embeddings of shape (batch, seq_len, embed_dim).
        """
        # Rotate and entangle the inputs
        rot = rotation_params.reshape(self.embed_dim, -1)
        ent = entangle_params.reshape(self.embed_dim, -1)
        rotated = torch.matmul(inputs, rot)
        entangled = torch.matmul(inputs, ent)

        # Project to query, key, value spaces
        Q = self.q_proj(rotated)
        K = self.k_proj(entangled)
        V = self.v_proj(inputs)

        # Compute attention scores via RBF kernel
        scores = self._rbf_kernel(Q, K)  # (batch, seq_len, seq_len)

        # Normalise scores
        attn = torch.softmax(scores, dim=-1)

        # Weighted sum of values
        return torch.matmul(attn, V)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute the Gram matrix between two sets of embeddings using the
        RBF kernel.  This mirrors the quantum kernel routine in the QML
        counterpart and can be used for downstream kernel‑based methods.
        """
        return self._rbf_kernel(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)

__all__ = ["HybridSelfAttention"]
