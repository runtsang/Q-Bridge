"""Pure‑classical self‑attention module with a hybrid sigmoid head.

This module implements a classical self‑attention block that mirrors the
interface of the original quantum‑inspired `SelfAttention` but remains
entirely NumPy/Torch based.  A lightweight `Hybrid` head provides a
differentiable sigmoid‑shifted output that can be dropped into
classification pipelines.

The class can be instantiated as:

    sa = UnifiedSelfAttentionHybrid(embed_dim=4)

and called as:

    out = sa.run(rotation_params, entangle_params, inputs)

where `rotation_params` and `entangle_params` are flat arrays of size
`3*embed_dim` and `embed_dim-1` respectively, and `inputs` is a
`(batch, embed_dim)` NumPy array.
"""

import numpy as np
import torch
import torch.nn as nn

class Hybrid(nn.Module):
    """Simple sigmoid‑shifted head used for classification."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.linear(inputs)
        return torch.sigmoid(logits + self.shift)

class UnifiedSelfAttentionHybrid:
    """Classical self‑attention with a hybrid sigmoid output."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim
        self.hybrid = Hybrid(in_features=embed_dim, shift=0.0)

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Compute self‑attention scores and apply a hybrid sigmoid head.

        Parameters
        ----------
        rotation_params : np.ndarray
            Flat array of length 3*embed_dim containing rotation angles.
        entangle_params : np.ndarray
            Flat array of length embed_dim-1 containing entanglement angles.
        inputs : np.ndarray
            Input tensor of shape (batch, embed_dim).

        Returns
        -------
        np.ndarray
            Output of shape (batch, 2) containing a probability pair
            (p, 1‑p) for binary classification.
        """
        query = inputs @ rotation_params.reshape(self.embed_dim, -1)
        key   = inputs @ entangle_params.reshape(self.embed_dim, -1)
        scores = np.exp(query @ key.T / np.sqrt(self.embed_dim))
        scores = scores / scores.sum(axis=-1, keepdims=True)
        weighted = scores @ inputs

        logits = torch.tensor(weighted, dtype=torch.float32)
        probs = self.hybrid(logits)
        probs = probs.detach().numpy()
        return np.concatenate((probs, 1 - probs), axis=-1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(embed_dim={self.embed_dim})"

__all__ = ["UnifiedSelfAttentionHybrid", "Hybrid"]
