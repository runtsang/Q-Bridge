"""Hybrid classical transformer that can optionally use a trainable self‑attention block.

The module is intentionally lightweight: it keeps the original
TextClassifier API but adds a ``use_self_attention`` flag.  When ``True``
the attention sub‑module is replaced by a *ClassicalSelfAttentionWrapper*,
which internally uses the SelfAttention helper from the seed
``SelfAttention.py``.  The wrapper exposes the same ``forward`` signature
as a typical Multi‑Head Attention layer so that the rest of the
architecture remains unchanged.

The design mirrors the quantum counterpart in the QML module, making
the two branches fully interchangeable from a client perspective.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the classical SelfAttention helper
from.SelfAttention import SelfAttention


class ClassicalSelfAttentionWrapper(nn.Module):
    """Self‑attention block that mimics the API of the quantum SelfAttention.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the token embeddings.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Trainable rotation / entangle matrices (square for simplicity)
        self.rotation_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, s, e = x.shape
        inp = x.reshape(b * s, e).cpu().numpy()
        rot_np = self.rotation_params.detach().cpu().numpy()
        ent_np = self.entangle_params.detach().cpu().numpy()
        out = SelfAttention().run(rot_np, ent_np, inp)
        return torch.from_numpy(out).to(x.device).reshape(b, s, e)


class FeedForwardClassical(nn.Module):
    """Two‑layer MLP used in the transformer block."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockHybrid(nn.Module):
    """Transformer block that can swap between multi‑head and self‑attention."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        attention_module: nn.Module,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = attention_module
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TextClassifierHybrid(nn.Module):
    """Transformer‑based text classifier with optional self‑attention."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_self_attention: bool = False,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        # Choose attention module
        if use_self_attention:
            attn_module = ClassicalSelfAttentionWrapper(embed_dim)
        else:
            from torch.nn import MultiheadAttention
            class MultiHeadAttentionWrapper(nn.Module):
                def __init__(self, embed_dim, num_heads, dropout):
                    super().__init__()
                    self.attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
                def forward(self, x, mask=None):
                    attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
                    return attn_output
            attn_module = MultiHeadAttentionWrapper(embed_dim, num_heads, dropout)

        self.transformers = nn.Sequential(
            *[
                TransformerBlockHybrid(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    dropout,
                    attention_module=attn_module,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


__all__ = [
    "ClassicalSelfAttentionWrapper",
    "FeedForwardClassical",
    "TransformerBlockHybrid",
    "PositionalEncoder",
    "TextClassifierHybrid",
]
