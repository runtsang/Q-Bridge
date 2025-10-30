from __future__ import annotations

import torch
import numpy as np
from torch import nn
from typing import Tuple

class SelfAttentionAutoencoder(nn.Module):
    """
    Hybrid classical self‑attention + auto‑encoder.

    The module first applies a scaled‑dot‑product self‑attention block
    (query/key/value) and then passes the attended representation
    through a fully‑connected auto‑encoder identical to the one in
    Autoencoder.py.  This mirrors the quantum interface while
    providing a rich classical baseline.
    """

    def __init__(
        self,
        embed_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Self‑attention parameters
        self.query_weights = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.key_weights   = nn.Parameter(torch.randn(embed_dim, embed_dim))

        # Auto‑encoder
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], embed_dim),
        )

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute scaled‑dot‑product self‑attention.
        """
        q = x @ self.query_weights
        k = x @ self.key_weights
        v = x
        scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self‑attention followed by auto‑encoder reconstruction.
        """
        attn = self.attention(x)
        latent = self.encoder(attn)
        recon = self.decoder(latent)
        return recon

__all__ = ["SelfAttentionAutoencoder"]
