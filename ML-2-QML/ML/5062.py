"""
Hybrid classical kernel module combining auto‑encoding, self‑attention
and an RBF kernel.  The implementation is deliberately lightweight
to keep training fast while still exposing the essential ideas from
the reference seeds.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence, List


# --------------------------------------------------------------------------- #
# 1. Utility: simple auto‑encoder that mirrors the seed AutoencoderNet
# --------------------------------------------------------------------------- #
@dataclass
class SimpleAutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class SimpleAutoencoder(nn.Module):
    """A lightweight MLP auto‑encoder used for dimensionality reduction."""

    def __init__(self, config: SimpleAutoencoderConfig) -> None:
        super().__init__()
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))


# --------------------------------------------------------------------------- #
# 2. Utility: simple self‑attention inspired by the seed SelfAttention
# --------------------------------------------------------------------------- #
class SimpleSelfAttention(nn.Module):
    """Self‑attention block that operates over a batch of latent vectors."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # learnable query, key and value matrices
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (batch, embed_dim)
        returns: attended representation of the same shape
        """
        queries = self.query_proj(inputs)
        keys = self.key_proj(inputs)
        values = self.value_proj(inputs)

        scores = torch.softmax(
            queries @ keys.transpose(-1, -2) / self.embed_dim**0.5, dim=-1
        )
        return scores @ values


# --------------------------------------------------------------------------- #
# 3. Classical RBF kernel
# --------------------------------------------------------------------------- #
class RBFKernel(nn.Module):
    """Radial Basis Function kernel with a learnable gamma."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x, y: (batch, dim)
        returns: (batch, 1) similarity scores
        """
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


def kernel_matrix(
    a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0
) -> torch.Tensor:
    """Convenience wrapper that builds an RBF kernel matrix."""
    kernel = RBFKernel(gamma)
    return torch.stack(
        [
            torch.stack([kernel(x, y) for y in b]).squeeze(-1)
            for x in a
        ]
    )


# --------------------------------------------------------------------------- #
# 4. HybridKernelMethod: orchestration of the above pieces
# --------------------------------------------------------------------------- #
class HybridKernelMethod(nn.Module):
    """
    Combines a simple auto‑encoder, self‑attention, and an RBF kernel.
    The class is designed to be a drop‑in replacement for the original
    QuantumKernelMethod, while adding richer preprocessing.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        gamma: float = 1.0,
        use_autoencoder: bool = True,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.use_autoencoder = use_autoencoder
        self.use_attention = use_attention

        if self.use_autoencoder:
            config = SimpleAutoencoderConfig(
                input_dim=input_dim, latent_dim=latent_dim
            )
            self.autoencoder = SimpleAutoencoder(config)

        if self.use_attention:
            # attention operates in the latent space
            self.attention = SimpleSelfAttention(embed_dim=latent_dim)

        self.kernel = RBFKernel(gamma=self.gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel between x and y after optional
        auto‑encoding and self‑attention.
        """
        if self.use_autoencoder:
            x = self.autoencoder.encode(x)
            y = self.autoencoder.encode(y)

        if self.use_attention:
            x = self.attention(x)
            y = self.attention(y)

        return self.kernel(x, y)

    def kernel_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the full kernel matrix between two collections of samples.
        """
        return torch.stack(
            [
                torch.stack([self.forward(x, y) for y in b]).squeeze(-1)
                for x in a
            ]
        )


__all__ = [
    "SimpleAutoencoder",
    "SimpleSelfAttention",
    "RBFKernel",
    "HybridKernelMethod",
    "kernel_matrix",
]
