"""Hybrid classical layer combining a fully connected network and self‑attention.

The class implements a forward pass that first projects the input into a
higher‑dimensional space with a linear layer and then refines it via
self‑attention (multi‑head, batch‑first).  An auxiliary ``run`` method
mirrors the original seed interfaces and accepts separate parameter sets
for the FC and attention parts, returning the sum of both contributions.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

class HybridLayer(nn.Module):
    def __init__(self, n_features: int = 1, embed_dim: int = 4, n_heads: int = 1) -> None:
        super().__init__()
        # Fully connected projection
        self.fc = nn.Linear(n_features, embed_dim)
        # Self‑attention block
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: linear projection → self‑attention.
        """
        z = torch.tanh(self.fc(x))
        attn_out, _ = self.attn(z, z, z)
        return attn_out

    def run(
        self,
        thetas: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Emulates the seed interface:
        * thetas   → linear weights
        * rotation_params, entangle_params → attention weights
        * inputs   → data matrix
        Returns the element‑wise sum of the FC and attention outputs.
        """
        # Linear part
        fc_weights = torch.as_tensor(thetas.reshape(-1, 1), dtype=torch.float32)
        fc_out = torch.tanh(torch.as_tensor(inputs, dtype=torch.float32) @ fc_weights).mean(dim=0).numpy()

        # Attention part
        rot_mat = torch.as_tensor(rotation_params.reshape(self.attn.embed_dim, -1), dtype=torch.float32)
        ent_mat = torch.as_tensor(entangle_params.reshape(self.attn.embed_dim, -1), dtype=torch.float32)
        query = torch.as_tensor(inputs, dtype=torch.float32) @ rot_mat
        key   = torch.as_tensor(inputs, dtype=torch.float32) @ ent_mat
        value = torch.as_tensor(inputs, dtype=torch.float32)

        scores = torch.softmax(query @ key.T / np.sqrt(self.attn.embed_dim), dim=-1)
        attn_out = (scores @ value).numpy()

        return fc_out + attn_out

__all__ = ["HybridLayer"]
