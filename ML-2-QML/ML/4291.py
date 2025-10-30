"""Hybrid classifier that fuses a classical autoencoder with a linear head.

This module defines :class:`HybridClassifierAutoencoder`, a PyTorch
``nn.Module`` that first compresses the input into a latent vector
using a fully‑connected autoencoder, then applies a lightweight linear
classifier.  The design follows the *combination* scaling paradigm
by keeping a deep classical encoder and a shallow quantum‑style
classifier interface.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple

from.Autoencoder import Autoencoder

__all__ = ["HybridClassifierAutoencoder"]

class HybridClassifierAutoencoder(nn.Module):
    """A classical hybrid classifier that mirrors the quantum helper interface.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the raw input vector.
    latent_dim : int, default 32
        Size of the latent space produced by the autoencoder.
    hidden_dims : Tuple[int,...], default (128, 64)
        Hidden layer sizes for the autoencoder encoder/decoder.
    dropout : float, default 0.1
        Drop‑out probability in the autoencoder.
    depth : int, default 3
        Depth of the quantum‑style classifier head (kept for API compatibility).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int,...] = (128, 64),
        dropout: float = 0.1,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = Autoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        # A shallow linear head that emulates the quantum classifier output.
        self.classifier = nn.Linear(latent_dim, 2)
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and linear head."""
        latent = self.encoder.encode(x)
        logits = self.classifier(latent)
        return logits
