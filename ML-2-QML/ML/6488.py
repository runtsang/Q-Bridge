"""
ml_self_attention_hybrid.py

Provides a classical self‑attention module with a regression head,
inspired by the SelfAttention and EstimatorQNN seeds.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SelfAttentionHybrid(nn.Module):
    """Hybrid classical self‑attention + regression network.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    hidden_dim : int, optional
        Size of the hidden layer in the regression head.
    """

    def __init__(self, embed_dim: int = 4, hidden_dim: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.attn = nn.Linear(embed_dim, embed_dim, bias=False)

        # Regression head (mirroring EstimatorQNN architecture)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        """
        Compute self‑attention scores from rotation and entangle parameters,
        then feed the attended representation through a regression head.

        Parameters
        ----------
        inputs : torch.Tensor
            Input matrix of shape (batch, embed_dim).
        rotation_params : np.ndarray
            Array of shape (embed_dim * 3,) used to construct the query matrix.
        entangle_params : np.ndarray
            Array of shape (embed_dim * 3,) used to construct the key matrix.

        Returns
        -------
        torch.Tensor
            Predicted scalar for each batch element.
        """
        # Build query and key matrices from parameters
        Q = torch.from_numpy(
            inputs @ rotation_params.reshape(self.embed_dim, -1)
        ).float()
        K = torch.from_numpy(
            inputs @ entangle_params.reshape(self.embed_dim, -1)
        ).float()

        # Attention scores
        scores = torch.softmax(Q @ K.t() / np.sqrt(self.embed_dim), dim=-1)

        # Attended values
        V = inputs.float()
        attended = scores @ V

        # Regression head
        return self.regressor(attended)

    def run(self, *args, **kwargs):
        """Compatibility method for the original interface."""
        return self.forward(*args, **kwargs)


__all__ = ["SelfAttentionHybrid"]
