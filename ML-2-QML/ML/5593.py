"""Unified classical QCNN combining autoencoding, graph pooling, and transformer attention.

The architecture mirrors the quantum counterpart: an autoencoder reduces dimensionality,
a fidelity‑based graph captures similarity between latent states, and a stack of
classical transformer blocks aggregates information across the graph. The final
fully‑connected head produces logits.  The class is intentionally lightweight
yet demonstrates how classical and quantum ideas can be combined.
"""

from __future__ import annotations

import torch
from torch import nn
from.Autoencoder import AutoencoderNet, AutoencoderConfig
from.GraphQNN import fidelity_adjacency
from.QTransformerTorch import TransformerBlockClassical

__all__ = ["UnifiedQCNN"]


class UnifiedQCNN(nn.Module):
    """Hybrid classical QCNN with autoencoder, graph pooling, and transformer blocks."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 64,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        # Auto‑encoding stage
        self.autoencoder = AutoencoderNet(
            AutoencoderConfig(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
            )
        )
        # Transformer stack that operates on graph‑pooled latent vectors
        self.transformer = nn.Sequential(
            *[
                TransformerBlockClassical(
                    embed_dim=latent_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=0.1,
                )
                for _ in range(num_blocks)
            ]
        )
        # Final classification/regression head
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.Tanh(),
            nn.Linear(16, num_classes),
        )

    # -------------------------------------------------------------------------
    # Helper utilities
    # -------------------------------------------------------------------------
    def _graph_pool(self, latent: torch.Tensor, threshold: float = 0.9) -> torch.Tensor:
        """Average each latent vector with its high‑fidelity neighbours."""
        n = latent.size(0)
        # Compute pairwise fidelities
        norms = torch.norm(latent, dim=1, keepdim=True) + 1e-12
        normed = latent / norms
        sims = torch.mm(normed, normed.t())
        # Build adjacency mask
        mask = sims >= threshold
        pooled = []
        for i in range(n):
            neighs = mask[i].nonzero(as_tuple=False).squeeze()
            if neighs.numel() > 0:
                pooled.append(latent[i] + latent[neighs].mean(0))
            else:
                pooled.append(latent[i])
        return torch.stack(pooled)

    # -------------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Encode input to latent space
        latent = self.autoencoder.encode(x)
        # Graph‑based pooling
        pooled = self._graph_pool(latent)
        # Transformer aggregation
        transformed = self.transformer(pooled.unsqueeze(0))  # add batch dim
        # Classification head
        out = self.head(transformed.mean(dim=1))
        return out.squeeze()
