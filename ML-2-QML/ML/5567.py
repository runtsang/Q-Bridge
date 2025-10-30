"""Hybrid sampler network combining transformer, autoencoder, and RBF kernel.

The module exposes a classical implementation `HybridSamplerQNN` that
mirrors the quantum helper while providing feature extraction via a
transformer stack and an autoencoder.  The API is fully compatible with
the original `SamplerQNN.py`, enabling easy drop‑in replacement.

The architecture is deliberately lightweight:
* Token embeddings + positional encoding → transformer encoder
* Sequence mean → autoencoder encoder (latent space)
* Latent → linear layer → softmax over two classes
* Classical RBF kernel utilities are also provided.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Configuration and helpers
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Configuration for the inner autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


# --------------------------------------------------------------------------- #
#  Core building blocks
# --------------------------------------------------------------------------- #
class AutoencoderNet(nn.Module):
    """A simple fully‑connected autoencoder used as a feature extractor."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers: list[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
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
        """Return the latent representation."""
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstruct from the latent space."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding used by the transformer stack."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    """Standard multi‑head attention used inside the transformer."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class FeedForward(nn.Module):
    """Two‑layer feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerEncoder(nn.Module):
    """Stack of transformer blocks used as a feature extractor."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


# --------------------------------------------------------------------------- #
#  Hybrid sampler network
# --------------------------------------------------------------------------- #
class HybridSamplerQNN(nn.Module):
    """
    Hybrid sampler that combines a transformer encoder, an autoencoder, and a
    final softmax layer.  The architecture is intentionally lightweight to keep
    training fast while still providing expressive feature extraction.

    Parameters
    ----------
    vocab_size : int
        Size of the input token vocabulary.
    embed_dim : int
        Dimensionality of the token embeddings.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Dimensionality of the feed‑forward sub‑network.
    latent_dim : int
        Size of the autoencoder latent space.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer = TransformerEncoder(embed_dim, num_heads, num_blocks, ffn_dim, dropout)

        # Autoencoder that consumes the *flattened* transformer output
        self.autoencoder = AutoencoderNet(
            AutoencoderConfig(
                input_dim=embed_dim,
                latent_dim=latent_dim,
                hidden_dims=(ffn_dim, embed_dim),
                dropout=dropout,
            )
        )

        # Final linear layer mapping latent space to 2 classes
        self.classifier = nn.Linear(latent_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Token embeddings + positional encoding
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)

        # Transformer feature extraction
        x = self.transformer(x)          # shape (batch, seq, embed_dim)

        # Collapse sequence dimension (mean) and feed into autoencoder
        x = x.mean(dim=1)                # shape (batch, embed_dim)
        latent = self.autoencoder.encode(x)  # shape (batch, latent_dim)

        logits = self.classifier(latent)
        return F.softmax(logits, dim=-1)


# --------------------------------------------------------------------------- #
#  Classical kernel utilities
# --------------------------------------------------------------------------- #
class RBFKernel(nn.Module):
    """Radial basis function kernel implementation."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


def kernel_matrix(a: Iterable[torch.Tensor], b: Iterable[torch.Tensor], gamma: float = 1.0) -> torch.Tensor:
    """Compute the Gram matrix between two collections of tensors."""
    kernel = RBFKernel(gamma)
    return torch.stack(
        [torch.cat([kernel(x, y) for y in b]) for x in a]
    ).squeeze()


# --------------------------------------------------------------------------- #
#  Factory helpers
# --------------------------------------------------------------------------- #
def HybridSamplerQNNFactory(
    vocab_size: int,
    embed_dim: int,
    num_heads: int,
    num_blocks: int,
    ffn_dim: int,
    latent_dim: int = 32,
    dropout: float = 0.1,
) -> HybridSamplerQNN:
    """
    Factory that returns a fully‑configured hybrid sampler model.
    The function mirrors the original `SamplerQNN` API to ease integration.
    """
    return HybridSamplerQNN(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        ffn_dim=ffn_dim,
        latent_dim=latent_dim,
        dropout=dropout,
    )


__all__ = [
    "AutoencoderConfig",
    "AutoencoderNet",
    "HybridSamplerQNN",
    "HybridSamplerQNNFactory",
    "RBFKernel",
    "kernel_matrix",
]
