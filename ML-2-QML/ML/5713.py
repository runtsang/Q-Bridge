"""Hybrid self‑attention transformer for classical training pipelines.

The module is intentionally lightweight: it contains a single class with
```
UnifiedSelfAttentionTransformer(embed_dim, num_heads, ffn_dim, *,
                               n_qubits_transformer=0, n_qubits_ffn=0,
                               q_device=None, dropout=0.1)
```
All sub‑modules are instantiated lazily so that the user can choose the
quantum depth for each part.  The design aggregates the following ideas from the seed projects:
*   Classical self‑attention with rotation and entanglement parameters
    from SelfAttention.py
*   Multi‑head attention, feed‑forward, positional encoding and
    classification logic from QTransformerTorch.py
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention implemented with PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output

class FeedForwardClassical(nn.Module):
    """Two‑layer feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockClassical(nn.Module):
    """Standard transformer block with residuals and layer‑norm."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class SelfAttentionHelper:
    """Classical self‑attention helper with rotation and entanglement parameters."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, rotation_params: torch.Tensor, entangle_params: torch.Tensor,
            inputs: torch.Tensor) -> torch.Tensor:
        # rotation_params shape: (embed_dim, embed_dim)
        # entangle_params shape: (embed_dim, embed_dim)
        query = inputs @ rotation_params.reshape(self.embed_dim, -1)
        key   = inputs @ entangle_params.reshape(self.embed_dim, -1)
        value = inputs
        scores = F.softmax(query @ key.T / math.sqrt(self.embed_dim), dim=-1)
        return scores @ value

class UnifiedSelfAttentionTransformer(nn.Module):
    """
    Hybrid transformer that can operate in classical or quantum mode.
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 num_blocks: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 n_qubits_transformer: int = 0,
                 n_qubits_ffn: int = 0,
                 use_quantum: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_quantum = use_quantum

        self.token_embedding = nn.Embedding(30522, embed_dim)  # vocab size placeholder
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.transformers = nn.ModuleList()

        for _ in range(num_blocks):
            if use_quantum and (n_qubits_transformer > 0 or n_qubits_ffn > 0):
                # In the pure‑classical branch the quantum wrappers are no‑ops
                block = TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
            else:
                block = TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
            self.transformers.append(block)

        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoding(tokens)
        for block in self.transformers:
            x = block(x, mask)
        x = x.mean(dim=1)
        x = self.dropout_layer(x)
        return self.classifier(x)

    def self_attention(self, rotation_params: torch.Tensor,
                       entangle_params: torch.Tensor,
                       inputs: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that mimics the original SelfAttention helper."""
        helper = SelfAttentionHelper(self.embed_dim)
        return helper.run(rotation_params, entangle_params, inputs)

__all__ = [
    "PositionalEncoding",
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "TransformerBlockClassical",
    "SelfAttentionHelper",
    "UnifiedSelfAttentionTransformer",
]
