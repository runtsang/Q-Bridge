"""Hybrid self‑attention layer combining classical attention with a quantum‑style fully‑connected layer.

The class implements the same API as the original SelfAttention seed, but replaces the value
projection with a small PyTorch linear layer that mimics the behaviour of the
parameter‑ised quantum circuit in the QML seed.  The module is fully
stand‑alone and can be used in any PyTorch workflow.

Usage
------
>>> from SelfAttention__gen075 import HybridSelfAttention
>>> layer = HybridSelfAttention(embed_dim=4, n_features=4)
>>> output = layer.run(rotation_params, entangle_params, inputs)
"""

import numpy as np
import torch
from torch import nn
from typing import Iterable

class HybridSelfAttention(nn.Module):
    """
    Classical self‑attention with a fully‑connected value extractor.
    """
    def __init__(self, embed_dim: int, n_features: int = 4):
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of the input embeddings.
        n_features : int, optional
            Size of the linear layer used to produce the value vector.
            Defaults to 4 to match the quantum circuit in the seed.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        # Value extractor – mimics the quantum FCL
        self.value = nn.Linear(n_features, embed_dim, bias=False)

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass of the hybrid attention layer.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters used to compute the query matrix.
        entangle_params : np.ndarray
            Parameters used to compute the key matrix.
        inputs : np.ndarray
            Input batch of shape (batch, embed_dim).

        Returns
        -------
        np.ndarray
            The attention‑weighted value tensor of shape (batch, embed_dim).
        """
        # Convert to tensors
        inp = torch.as_tensor(inputs, dtype=torch.float32)

        # Compute query and key projections using the provided parameters
        q = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        k = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)

        # Attention scores
        scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)

        # Value projection – use the input itself as features for the linear layer
        val = self.value(inp)

        # Weighted sum
        out = torch.matmul(scores, val)

        return out.detach().cpu().numpy()

__all__ = ["HybridSelfAttention"]
