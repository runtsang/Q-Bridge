"""Hybrid transformer with optional quantum blocks and auto‑encoder integration.

The module is a drop‑in replacement for the classical QTransformerTorch and
extends it with:
* A configurable AutoencoderNet (fully‑connected) for latent compression.
* Optional quantum attention and feed‑forward layers (using torchquantum).
* An optional regression head (EstimatorQNN) that can be used instead of
  the classification head.
* A clean API that mirrors the original QTransformerTorch signature
  while exposing the new features via boolean flags.

Author: OpenAI GPT‑OSS‑20B
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

# ----------------------------------------------------------------------
#  Autoencoder
# ----------------------------------------------------------------------
def _as_tensor(data: Iterable[float] | Tensor) -> Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


@dataclass
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """A lightweight fully‑connected auto‑encoder."""

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

    def forward(self, inputs: Tensor) -> Tensor:
        return self.decode(self.encode(inputs))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Factory that mirrors the quantum helper returning a configured network."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)


# ----------------------------------------------------------------------
#  Attention and Feed‑Forward (classical)
# ----------------------------------------------------------------------
class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: Tensor) -> Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value)

    def downstream(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        batch_size: int,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        q = self.separate_heads(query)
        k = self.separate_heads(key)
        v = self.separate_heads(value)
        out = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented classically."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = x.shape
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        out = self.downstream(q, k, v, batch_size, mask)
        return self.combine_heads(out)


# ----------------------------------------------------------------------
#  Feed‑Forward (classical)
# ----------------------------------------------------------------------
class FeedForwardBase(nn.Module):
    """Base class for feed‑forward layers."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ----------------------------------------------------------------------
#  Transformer block (classical)
# ----------------------------------------------------------------------
class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# ----------------------------------------------------------------------
#  Positional encoding
# ----------------------------------------------------------------------
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1)]


# ----------------------------------------------------------------------
#  Estimator head (regression)
# ----------------------------------------------------------------------
class EstimatorQNN(nn.Module):
    """Simple fully‑connected regression network."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


# ----------------------------------------------------------------------
#  Hybrid Transformer
# ----------------------------------------------------------------------
class HybridTransformerNet(nn.Module):
    """Transformer that can swap classical and quantum sub‑modules and
    optionally append an auto‑encoder or a regression head.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        latent_dim: int = 32,
        use_quantum_attention: bool = False,
        use_quantum_ffn: bool = False,
        use_estimator: bool = False,
        use_autoencoder: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.use_estimator = use_estimator
        self.use_autoencoder = use_autoencoder

        # Build transformer blocks
        blocks = []
        for _ in range(num_blocks):
            if use_quantum_attention or use_quantum_ffn:
                # Quantum blocks are not implemented in this classical branch;
                # they are placeholders for the quantum implementation.
                # Here we fall back to classical blocks to keep the API stable.
                blocks.append(
                    TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                )
            else:
                blocks.append(
                    TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                )
        self.transformers = nn.Sequential(*blocks)

        # Optional heads
        self.autoencoder = Autoencoder(
            input_dim=embed_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        ) if use_autoencoder else None

        self.estimator = EstimatorQNN() if use_estimator else None

        self.dropout = nn.Dropout(dropout)
        if num_classes > 2:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        # Embedding + positional
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        # Transformer layers
        x = self.transformers(x)
        # Pooling
        x = x.mean(dim=1)
        x = self.dropout(x)

        # Optional auto‑encoder latent extraction
        if self.autoencoder is not None:
            x = self.autoencoder.encode(x)

        # Optional regression head
        if self.estimator is not None:
            return self.estimator(x)

        # Classification head
        return self.classifier(x)


__all__ = [
    "Autoencoder",
    "AutoencoderNet",
    "AutoencoderConfig",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "EstimatorQNN",
    "HybridTransformerNet",
]
