"""
Hybrid self‑attention module that embeds a classical attention
mechanism with a small quantum‑inspired fully‑connected layer.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

class HybridSelfAttention(nn.Module):
    """
    Classical self‑attention layer augmented with a quantum
    fully‑connected transformation.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    n_features : int, optional
        Size of the intermediate fully‑connected layer.
    """
    def __init__(self, embed_dim: int, n_features: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_features = n_features

        # Linear maps for query, key, value
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Quantum‑inspired fully‑connected layer (implemented classically)
        self.fc = nn.Linear(n_features, 1, bias=False)

    def forward(self,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray,
                inputs: np.ndarray) -> np.ndarray:
        """
        Compute the hybrid self‑attention output.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for generating query vectors.
        entangle_params : np.ndarray
            Parameters for generating key vectors.
        inputs : np.ndarray
            Input embedding matrix of shape (batch, embed_dim).

        Returns
        -------
        np.ndarray
            Attention‑weighted output of shape (batch, embed_dim).
        """
        x = torch.as_tensor(inputs, dtype=torch.float32, device=self.query_proj.weight.device)

        # Classical query/key/value
        query = self.query_proj(x)
        key   = self.key_proj(x)
        value = self.value_proj(x)

        # Classical attention scores
        scores = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)

        # Quantum‑inspired feature: feed a subset of the scores into the
        # fully‑connected layer to introduce non‑linear dependency.
        # We take a mean over the key dimension to obtain a 1‑D vector.
        q_features = scores.mean(dim=-1)  # shape (batch,)
        q_features = q_features.unsqueeze(-1)  # (batch, 1)

        # Map to intermediate features
        q_features = torch.tanh(self.fc(q_features))  # (batch, 1)

        # Modulate the classical attention output
        output = torch.matmul(scores, value) * q_features

        return output.detach().numpy()

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """Compatibility wrapper for the seed interface."""
        return self.forward(rotation_params, entangle_params, inputs)

__all__ = ["HybridSelfAttention"]
