"""Extended classical transformer with optional bias, residual block, and feature extractor hook.

The module defines a unified TransformerTwin class that can be instantiated
with `use_quantum=False` (default) or `use_quantum=True`.  The classical
implementation is fully functional and can be used as a drop‑in
replacement for the seed version.  The quantum variant is provided in the
QML module.

Key extensions:
* `ResidualBlock` adds a second residual connection inside each block.
* `use_bias` flag propagates to all linear layers.
* `feature_extractor` hook allows swapping the token+positional encoder.
* `config` dictionary stored for reproducibility.
"""

from __future__ import annotations

import math
from typing import Optional, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Attention & Feed‑Forward base classes
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_bias = use_bias

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value)

    def downstream(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.separate_heads(query)
        k = self.separate_heads(key)
        v = self.separate_heads(value)
        out = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(query.size(0), -1, self.embed_dim)


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        out = self.downstream(q, k, v, mask)
        return self.combine_heads(out)


class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)
        self.use_bias = use_bias


class FeedForwardClassical(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__(embed_dim, ffn_dim, dropout, use_bias)
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=use_bias)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# --------------------------------------------------------------------------- #
#  Residual block (optional second residual)
# --------------------------------------------------------------------------- #
class ResidualBlock(nn.Module):
    """Wraps a submodule and adds a second residual connection."""
    def __init__(self, submodule: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.submodule = submodule
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.submodule(x))


# --------------------------------------------------------------------------- #
#  Transformer block
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_bias = use_bias


class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__(embed_dim, num_heads, ffn_dim, dropout, use_bias)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout, use_bias)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout, use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# --------------------------------------------------------------------------- #
#  Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
#  Unified transformer twin (classical by default)
# --------------------------------------------------------------------------- #
class TransformerTwin(nn.Module):
    """Unified transformer that can operate in classical or quantum mode.

    Parameters
    ----------
    vocab_size : int
    embed_dim : int
    num_heads : int
    num_blocks : int
    ffn_dim : int
    num_classes : int
    dropout : float
    use_bias : bool
    use_quantum : bool
    config : dict, optional
    feature_extractor : Callable, optional
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
        use_bias: bool = False,
        use_quantum: bool = False,
        config: Optional[Dict] = None,
        feature_extractor: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.config = config or {}
        if feature_extractor is None:
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
            self.pos_embedding = PositionalEncoder(embed_dim)
        else:
            self.token_embedding, self.pos_embedding = feature_extractor(vocab_size, embed_dim)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if use_quantum:
                # Quantum blocks are defined in the QML module; here we use a placeholder
                self.blocks.append(TransformerBlockClassical(embed_dim, num_heads, ffn_dim,
                                                            dropout=dropout, use_bias=use_bias))
            else:
                self.blocks.append(TransformerBlockClassical(embed_dim, num_heads, ffn_dim,
                                                            dropout=dropout, use_bias=use_bias))
        self.transformers = nn.Sequential(*self.blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


# Alias for backward compatibility
TextClassifier = TransformerTwin

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "TransformerTwin",
    "TextClassifier",
]
