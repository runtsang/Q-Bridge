"""Combined transformer with optional autoencoder and convolutional feature extractor (classical)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Iterable

import torch
from torch import nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Autoencoder
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Configuration for a small fully‑connected autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Lightweight fully‑connected autoencoder."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers: list[nn.Module] = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(cfg)


# --------------------------------------------------------------------------- #
# Convolutional filter (classical)
# --------------------------------------------------------------------------- #
def Conv() -> nn.Module:
    """Return a simple 2‑D convolutional layer that mimics a quantum filter."""
    class ConvFilter(nn.Module):
        def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        def run(self, data: Iterable[float]) -> float:
            tensor = torch.as_tensor(data, dtype=torch.float32)
            tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
            logits = self.conv(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean().item()

    return ConvFilter()


# --------------------------------------------------------------------------- #
# Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
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


# --------------------------------------------------------------------------- #
# Classical transformer components
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out


class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


# --------------------------------------------------------------------------- #
# Unified transformer
# --------------------------------------------------------------------------- #
class CombinedTransformer(nn.Module):
    """
    A transformer that can optionally prepend a classical autoencoder and/or a
    convolutional feature extractor.  The core transformer layers are classical
    by default but can be swapped for quantum variants in the QML module.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        *,
        dropout: float = 0.1,
        use_autoencoder: bool = False,
        autoencoder_latent_dim: int = 32,
        autoencoder_hidden: Tuple[int, int] = (128, 64),
        autoencoder_dropout: float = 0.1,
        conv_kernel: int = 2,
        conv_threshold: float = 0.0,
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)

        # Optional autoencoder
        self.use_autoencoder = use_autoencoder
        if use_autoencoder:
            cfg = AutoencoderConfig(
                input_dim=embed_dim,
                latent_dim=autoencoder_latent_dim,
                hidden_dims=autoencoder_hidden,
                dropout=autoencoder_dropout,
            )
            self.autoencoder = AutoencoderNet(cfg)
            # Map latent dimension back to embed_dim if necessary
            if autoencoder_latent_dim!= embed_dim:
                self.latent_to_embed = nn.Linear(autoencoder_latent_dim, embed_dim)
            else:
                self.latent_to_embed = None
        else:
            self.autoencoder = None
            self.latent_to_embed = None

        # Optional convolutional filter
        self.use_conv = use_conv
        if use_conv:
            self.conv = Conv()
        else:
            self.conv = None

        # Transformer layers
        self.transformer = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)

        # Classifier
        self.classifier = (
            nn.Linear(embed_dim, num_classes)
            if num_classes > 2
            else nn.Linear(embed_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: token indices of shape (batch, seq_len)
        Returns:
            logits of shape (batch, num_classes) or (batch, 1) for binary
        """
        # Embedding + optional convolution
        if self.use_conv:
            # Apply conv filter to each token (treat token as scalar)
            conv_vals = torch.stack([self.conv.run([float(tok)]) for tok in x.unbind(0)], dim=0)
            # Broadcast conv values to match embedding dimension
            conv_embedding = conv_vals.unsqueeze(-1).expand(-1, -1, self.token_embedding.embedding_dim)
            tokens = self.token_embedding(x) + conv_embedding
        else:
            tokens = self.token_embedding(x)

        # Optional autoencoder preprocessing
        if self.use_autoencoder:
            batch, seq_len, _ = tokens.shape
            flat = tokens.reshape(batch * seq_len, -1)
            latent = self.autoencoder.encode(flat)
            latent = latent.reshape(batch, seq_len, -1)
            if self.latent_to_embed is not None:
                latent = self.latent_to_embed(latent)
            tokens = latent

        # Positional encoding
        x = self.pos_encoder(tokens)

        # Transformer blocks
        x = self.transformer(x)

        # Pooling + classifier
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "Conv",
    "PositionalEncoder",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "CombinedTransformer",
]
