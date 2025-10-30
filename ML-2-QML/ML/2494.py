"""Unified self‑attention and transformer for pure‑classical use.

The module is compatible with the original SelfAttention.py and
the QTransformerTorch.py API.  It exposes a single class
`UnifiedSelfAttentionTransformer` that implements a transformer block
with optional quantum sub‑modules.  The classical implementation is
fully functional and can be used as a drop‑in replacement for the
quantum block in any existing project.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Classical self‑attention helper (kept for API compatibility)
class ClassicalSelfAttention:
    def __init__(self, embed_dim: int = 4):
        self.embed_dim = embed_dim

    def forward(
        self,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        # rotation_params and entangle_params are ignored; kept for
        # compatibility with the quantum version.
        q = inputs @ rotation_params.reshape(self.embed_dim, -1)
        k = inputs @ entangle_params.reshape(self.embed_dim, -1)
        v = inputs
        scores = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.embed_dim), dim=-1)
        return scores @ v

def SelfAttention() -> ClassicalSelfAttention:
    """Return a classical self‑attention layer with the same API as the
    quantum implementation in the original repository."""
    return ClassicalSelfAttention(embed_dim=4)

# --------------------------------------------------------------------------- #
# Transformer components
# --------------------------------------------------------------------------- #

class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x)
        return out

class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class TextClassifier(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim,
                                       dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim,
                                    num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

# --------------------------------------------------------------------------- #
# Unified transformer that can optionally use quantum sub‑modules
# --------------------------------------------------------------------------- #

class UnifiedSelfAttentionTransformer(nn.Module):
    """
    A hybrid transformer that can be instantiated with either classical
    or quantum attention/FFN sub‑modules.  The constructor arguments
    mirror the original QTransformerTorch API, but the default is
    classical.  Quantum sub‑modules are not loaded in this file to keep
    the classical implementation lightweight.
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1):
        super().__init__()
        self.transformer = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim,
                                       dropout) for _ in range(num_blocks)]
        )
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim,
                                    num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

__all__ = [
    "ClassicalSelfAttention",
    "SelfAttention",
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "TextClassifier",
    "UnifiedSelfAttentionTransformer",
]
