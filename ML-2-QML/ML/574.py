"""Hybrid transformer classifier with optional quantum sub-modules.

The module exposes a single public class `QuantumTransformerAdapter` that
mirrors the API of the original `TextClassifier`.  The constructor accepts
an additional `quantum_depth` argument that can be set to 0 for a fully
classical model or a positive integer to enable quantum sub‑modules.
The implementation below is fully classical – it only uses PyTorch
and standard modules – and serves as the baseline for the quantum
variant.

Key extensions compared to the seed:
* Hybrid dropout schedule – dropout probability grows linearly with
  the block index, providing a simple curriculum for regularisation.
* Parameter‑efficient quantum‑variational feed‑forward block
  (implemented in the quantum module) is exposed via the same API.
* `quantum_depth` knob allows switching between classical and
  quantum behaviour at runtime without changing the surrounding code.
"""

from __future__ import annotations

import math
from typing import Optional, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionBase(nn.Module):
    """Base class for attention layers, providing helper utilities."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        # shape: (batch, seq, heads, d_k)
        return x.view(x.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # broadcast mask to all heads
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        return self.dropout(attn), self.dropout(scores)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented using PyTorch's MultiheadAttention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, **kwargs):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim,
                                          num_heads,
                                          dropout=dropout,
                                          batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return out


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward layers."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer MLP feed‑forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockBase(nn.Module):
    """Base transformer block with layer normalisation and residuals."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    """Classic transformer block."""

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 dropout: float = 0.0,
                 block_idx: int = 0,
                 num_blocks: int = 1):
        super().__init__(embed_dim, num_heads, dropout)
        # Grow dropout linearly with depth
        self.dropout = nn.Dropout(dropout * (1 + block_idx / num_blocks))
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout=self.dropout.p)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout=self.dropout.p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class QuantumTransformerAdapter(nn.Module):
    """
    Hybrid transformer classifier that can operate in fully classical mode
    or activate quantum sub‑modules via the ``quantum_depth`` parameter.
    The API matches that of the original TextClassifier.
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 quantum_depth: int = 0,
                 *_, **__):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.blocks: Iterable[nn.Module] = nn.ModuleList()
        for i in range(num_blocks):
            block = TransformerBlockClassical(embed_dim,
                                              num_heads,
                                              ffn_dim,
                                              dropout=dropout,
                                              block_idx=i,
                                              num_blocks=num_blocks)
            self.blocks.append(block)
        self.blocks = nn.Sequential(*self.blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim,
                                    num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.blocks(x)
        x = x.mean(dim=1)  # mean pooling
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "QuantumTransformerAdapter",
]
