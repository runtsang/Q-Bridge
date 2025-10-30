"""Hybrid classical self‑attention module.

The implementation mirrors the quantum interface so that
`SelfAttention` can be swapped seamlessly between the
classical and quantum back‑ends.  Parameters are
expressed as plain NumPy arrays to keep the API identical
to the original seed, but the computation is carried out
with PyTorch tensors for efficient GPU acceleration.
"""

import numpy as np
import torch
from torch import nn
from typing import Iterable

class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        # Linear projections for query/key/value
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Compute a classical self‑attention map.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape ``(embed_dim, embed_dim)`` – used to initialise
            the projection matrices.
        entangle_params : np.ndarray
            Unused in the classical version but kept for API
            compatibility; it is interpreted as a scaling factor.
        inputs : np.ndarray
            Input token matrix ``(seq_len, embed_dim)``.
        """
        # Initialise projections from rotation_params
        self.query.weight.data = torch.as_tensor(
            rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        self.key.weight.data   = torch.as_tensor(
            entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        # Forward pass
        Q = self.query(torch.as_tensor(inputs, dtype=torch.float32))
        K = self.key(torch.as_tensor(inputs, dtype=torch.float32))
        V = self.value(torch.as_tensor(inputs, dtype=torch.float32))
        scores = torch.softmax(Q @ K.transpose(-2, -1) / np.sqrt(self.embed_dim),
                               dim=-1)
        out = scores @ V
        return out.detach().numpy()

__all__ = ["SelfAttention"]
