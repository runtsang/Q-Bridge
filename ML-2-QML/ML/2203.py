"""Hybrid classical self‑attention + autoencoder.

The module defines a `SelfAttention` helper that can be used inside a PyTorch
autoencoder.  Parameters are exposed as learnable tensors so that the
attention can be trained jointly with the rest of the network.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    """Differentiable self‑attention block.

    Parameters
    ----------
    embed_dim : int
        Dimension of the input embeddings.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Rotation and entanglement parameters become learnable weights.
        self.rotation = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute attention over ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Attention‑weighted representation of shape (batch, seq_len, embed_dim).
        """
        # Linear projections
        q = x @ self.rotation
        k = x @ self.entangle
        v = x

        # Scaled dot‑product attention
        scores = F.softmax((q @ k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v


class AutoencoderNet(nn.Module):
    """Autoencoder that inserts a self‑attention layer after the first encoder block."""

    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: tuple[int, int] = (128, 64)) -> None:
        super().__init__()
        # Encoder
        encoder_layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        encoder_layers += [nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU()]
        encoder_layers += [SelfAttention(hidden_dims[1])]
        encoder_layers += [nn.Linear(hidden_dims[1], latent_dim)]
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = [nn.Linear(latent_dim, hidden_dims[1]), nn.ReLU()]
        decoder_layers += [nn.Linear(hidden_dims[1], hidden_dims[0]), nn.ReLU()]
        decoder_layers += [nn.Linear(hidden_dims[0], input_dim)]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

__all__ = ["SelfAttention", "AutoencoderNet"]
