"""Hybrid transformer with autoencoder and hybrid classification head (classical implementation)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Attention and feed‑forward primitives
# ----------------------------------------------------------------------
class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out

class FeedForwardBase(nn.Module):
    """Base class for the feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# ----------------------------------------------------------------------
# Transformer block
# ----------------------------------------------------------------------
class TransformerBlockBase(nn.Module):
    """Base transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    """Standard transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# ----------------------------------------------------------------------
# Positional encoding
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x + self.pe[:, : x.size(1)]

# ----------------------------------------------------------------------
# Autoencoder
# ----------------------------------------------------------------------
class AutoencoderNet(nn.Module):
    """Simple fully‑connected autoencoder for embedding reconstruction."""
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

# ----------------------------------------------------------------------
# Hybrid classification head
# ----------------------------------------------------------------------
class HybridClassifier(nn.Module):
    """Linear head with a sigmoid shift to emulate a quantum expectation."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = self.linear(x)
        return torch.sigmoid(logits + self.shift)

# ----------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------
class HybridTransformerAutoencoder(nn.Module):
    """Hybrid transformer that optionally uses a quantum attention block and
    includes an autoencoder reconstruction branch and a hybrid classification head."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        autoencoder_latent: int = 32,
        autoencoder_hidden: Tuple[int, int] = (128, 64),
        hybrid_shift: float = 0.0,
        use_quantum: bool = False,  # flag for future quantum extension
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        # Build transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.autoencoder = AutoencoderNet(embed_dim, autoencoder_latent, autoencoder_hidden, dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        self.hybrid_head = HybridClassifier(embed_dim, shift=hybrid_shift)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            LongTensor of shape (batch, seq_len) containing token indices.

        Returns
        -------
        logits : torch.Tensor
            Classification logits of shape (batch, num_classes) or (batch, 1).
        reconstruction : torch.Tensor
            Reconstructed embeddings of shape (batch, seq_len, embed_dim).
        """
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.dropout(x)
        # Classification
        cls_logits = self.classifier(x.mean(dim=1))
        # Hybrid head
        cls_hybrid = self.hybrid_head(x.mean(dim=1))
        logits = torch.cat((cls_logits, cls_hybrid), dim=-1) if self.classifier.out_features > 1 else cls_hybrid
        # Reconstruction
        reconstruction = self.autoencoder(x)
        return logits, reconstruction

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "AutoencoderNet",
    "HybridClassifier",
    "HybridTransformerAutoencoder",
]
