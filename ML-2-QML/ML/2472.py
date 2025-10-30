"""Hybrid self‑attention module that integrates a classical attention block with a fully‑connected layer."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

class HybridSelfAttention(nn.Module):
    """
    Classical self‑attention with a fully‑connected layer.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    n_features : int, default 1
        Size of the fully‑connected layer input.
    """
    def __init__(self, embed_dim: int, n_features: int = 1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.fc = nn.Linear(n_features, 1)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the hybrid attention output.

        The rotation and entangle parameters are interpreted as in a classical
        self‑attention block.  The fully‑connected layer is applied to each
        input vector to obtain a scalar weight that modulates the attention
        output.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the query/key projections.
        entangle_params : np.ndarray
            Parameters for the key/value projections.
        inputs : np.ndarray
            Input matrix of shape (batch, embed_dim).

        Returns
        -------
        np.ndarray
            The hybrid attention output of shape (batch, 1).
        """
        # Classical self‑attention
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        attn_out = scores @ torch.as_tensor(inputs, dtype=torch.float32)

        # Fully‑connected weighting
        weight = torch.as_tensor(
            self.fc(torch.as_tensor(inputs, dtype=torch.float32)),
            dtype=torch.float32,
        )
        # Broadcast weight to match attention output
        return (attn_out * weight).detach().numpy()

def SelfAttention():
    """
    Factory that returns a HybridSelfAttention instance with default
    configuration matching the original anchor module.
    """
    return HybridSelfAttention(embed_dim=4, n_features=1)

__all__ = ["SelfAttention"]
