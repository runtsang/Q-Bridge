"""Hybrid classical autoencoder integrating quanvolution and transformer."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

# --------------------------------------------------------------------------- #
# Quanvolution filter
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    """Simple 2‑D convolutional filter mimicking a quantum-inspired kernel."""
    def __init__(self) -> None:
        super().__init__()
        # 1 input channel → 4 feature maps, 2×2 kernel, stride 2
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        # Flatten to (batch, 4*14*14)
        return features.view(x.size(0), -1)

# --------------------------------------------------------------------------- #
# Transformer components (taken from QTransformerTorch)
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
# Hybrid Autoencoder
# --------------------------------------------------------------------------- #
class HybridAutoencoder(nn.Module):
    """Classical autoencoder that combines a quanvolution encoder, transformer
    feature extraction, and a linear decoder."""
    def __init__(
        self,
        input_channels: int = 1,
        input_size: int = 28,
        latent_dim: int = 32,
        transformer_heads: int = 4,
        transformer_ffn: int = 64,
        transformer_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.quanvolution = QuanvolutionFilter()
        self.input_dim = 4 * (input_size // 2) * (input_size // 2)  # 4 * 14 * 14 = 784

        # Embed each pixel to an embedding vector
        self.embed_dim = transformer_ffn
        self.embedding = nn.Linear(self.input_dim, self.input_dim * self.embed_dim)
        self.positional = PositionalEncoder(self.embed_dim)

        # Transformer encoder
        self.transformer = nn.Sequential(
            *[
                TransformerBlockClassical(self.embed_dim, transformer_heads, self.embed_dim, dropout)
                for _ in range(transformer_blocks)
            ]
        )

        # Projection to latent space
        self.latent_proj = nn.Linear(self.embed_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.input_dim),
            nn.ReLU(),
            nn.Unflatten(1, (1, input_size, input_size)),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder."""
        x = self.quanvolution(x)  # (batch, input_dim)
        x = self.embedding(x)  # (batch, input_dim * embed_dim)
        x = x.view(x.size(0), self.input_dim, self.embed_dim)  # (batch, seq_len, embed_dim)
        x = self.positional(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        latent = self.latent_proj(x)
        return latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


__all__ = [
    "HybridAutoencoder",
    "QuanvolutionFilter",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "PositionalEncoder",
]
