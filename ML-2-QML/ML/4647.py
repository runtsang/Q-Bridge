"""Hybrid classical self‑attention that incorporates an autoencoder.

The module defines a single class `SelfAttentionGen` that performs:
  1. Dimensionality reduction via a small MLP autoencoder.
  2. Self‑attention on the latent representation.
  3. Optional decoding back to the original feature space.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor


def _as_tensor(data):
    if isinstance(data, Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


@dataclasses.dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Simple fully‑connected autoencoder."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def Autoencoder(input_dim: int, *, latent_dim=32, hidden_dims=(128, 64), dropout=0.1) -> AutoencoderNet:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)


class SelfAttentionGen:
    """Classical self‑attention block with optional autoencoding."""

    def __init__(
        self,
        embed_dim: int,
        latent_dim: int = 32,
        *,
        decoder: bool = True,
    ) -> None:
        self.embed_dim = embed_dim
        self.decoder = decoder
        self.autoencoder = Autoencoder(embed_dim, latent_dim=latent_dim)
        self.latent_dim = latent_dim

    def _attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        scores = torch.softmax(query @ key.T / np.sqrt(self.latent_dim), dim=-1)
        return scores @ value

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : (latent_dim*3,) array
            Parameters for 3‑angle rotations per latent dimension.
        entangle_params : (latent_dim-1,) array
            Parameters for consecutive CX‑like entangling gates.
        inputs : (batch, embed_dim) array
            Raw input features.
        Returns
        -------
        output : np.ndarray
            Attention‑weighted representations, optionally decoded to original space.
        """
        x = _as_tensor(inputs)
        # Encode to latent space
        latents = self.autoencoder.encode(x)

        # Build query, key, value from latent representation
        q = torch.as_tensor(
            latents @ rotation_params.reshape(self.latent_dim, -1),
            dtype=torch.float32,
        )
        k = torch.as_tensor(
            latents @ entangle_params.reshape(self.latent_dim, -1),
            dtype=torch.float32,
        )
        v = latents

        attn = self._attention(q, k, v)

        if self.decoder:
            attn = self.autoencoder.decode(attn)

        return attn.numpy()


__all__ = ["SelfAttentionGen", "Autoencoder", "AutoencoderNet", "AutoencoderConfig"]
