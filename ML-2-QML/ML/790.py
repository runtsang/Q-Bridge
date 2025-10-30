"""
Hybrid transformer block for classical training with optional quantum sub‑modules.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Base attention and feed‑forward layers
# ----------------------------------------------------------------------
class MultiHeadAttentionBase(nn.Module):
    """
    Base class for all attention variants.
    Keeps the same API as the seed, but adds a *quantum‑aware* dropout
    that can be tuned during training.
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        # Learnable scaling for the dropout probability
        self.qdrop_scale = nn.Parameter(torch.tensor(dropout, dtype=torch.float32))
        self.use_bias = use_bias

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape into (batch, heads, seq, d_k)."""
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, query: torch.Tensor, key: torch.Tensor,
                  value: torch.Tensor, mask: Optional[torch.Tensor] = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot‑product attention with optional mask."""
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = mask.unsqueeze(1).masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """
    Standard multi‑head attention implemented classically.
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        attn_output, _ = self.attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_proj(attn_output)


class MultiHeadAttentionHybrid(MultiHeadAttentionBase):
    """
    Hybrid attention that applies a quantum module to each head after the
    classical projection and attention calculation.
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1, use_bias: bool = False,
                 quantum_module: Optional[nn.Module] = None) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.quantum_module = quantum_module or nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        attn_output, _ = self.attention(q, k, v, mask)
        # Apply quantum module to each head
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq, self.num_heads, self.d_k)
        heads = []
        for h in range(self.num_heads):
            head = attn_output[:, :, h, :]  # (batch, seq, d_k)
            head = self.quantum_module(head)
            heads.append(head)
        attn_output = torch.stack(heads, dim=2).contiguous()
        attn_output = attn_output.view(batch, seq, self.embed_dim)
        return self.out_proj(attn_output)


class FeedForwardBase(nn.Module):
    """
    Base feed‑forward network.
    """
    def __init__(self, embed_dim: int, ffn_dim: int,
                 dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)
        self.use_bias = use_bias


class FeedForwardClassical(FeedForwardBase):
    """
    Two‑layer perceptron feed‑forward network.
    """
    def __init__(self, embed_dim: int, ffn_dim: int,
                 dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__(embed_dim, ffn_dim, dropout, use_bias)
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=use_bias)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardHybrid(FeedForwardBase):
    """
    Feed‑forward network that optionally uses a quantum module after the first linear layer.
    """
    def __init__(self, embed_dim: int, ffn_dim: int,
                 dropout: float = 0.1, use_bias: bool = False,
                 quantum_module: Optional[nn.Module] = None) -> None:
        super().__init__(embed_dim, ffn_dim, dropout, use_bias)
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=use_bias)
        self.quantum_module = quantum_module or nn.Identity()
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.quantum_module(x)
        return self.linear2(self.dropout(F.relu(x)))


# ----------------------------------------------------------------------
# Transformer blocks
# ----------------------------------------------------------------------
class TransformerBlockBase(nn.Module):
    """
    Base transformer block containing attention and feed‑forward parts.
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_bias = use_bias


class TransformerBlockClassical(TransformerBlockBase):
    """
    Classical transformer block.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, use_bias: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads,
                                                dropout, use_bias)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim,
                                        dropout, use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockHybrid(TransformerBlockBase):
    """
    Hybrid transformer block that can mix classical and quantum sub‑modules.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, use_bias: bool = False,
                 quantum_attn_module: Optional[nn.Module] = None,
                 quantum_ffn_module: Optional[nn.Module] = None) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.attn = MultiHeadAttentionHybrid(embed_dim, num_heads,
                                             dropout, use_bias,
                                             quantum_module=quantum_attn_module)
        self.ffn = FeedForwardHybrid(embed_dim, ffn_dim,
                                     dropout, use_bias,
                                     quantum_module=quantum_ffn_module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# ----------------------------------------------------------------------
# Positional encoding
# ----------------------------------------------------------------------
class PositionalEncoder(nn.Module):
    """
    Sinusoidal positional encoding.
    """
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
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


# ----------------------------------------------------------------------
# Text classifier
# ----------------------------------------------------------------------
class TextClassifier(nn.Module):
    """
    Transformer‑based text classifier supporting hybrid quantum sub‑modules.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_bias: bool = False,
                 hybrid: bool = False,
                 quantum_attn_module: Optional[nn.Module] = None,
                 quantum_ffn_module: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        block_cls = TransformerBlockHybrid if hybrid else TransformerBlockClassical
        self.transformer_blocks = nn.Sequential(*[
            block_cls(embed_dim, num_heads, ffn_dim,
                      dropout, use_bias,
                      quantum_attn_module, quantum_ffn_module)
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim,
                                    num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        x = self.transformer_blocks(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionHybrid",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardHybrid",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockHybrid",
    "PositionalEncoder",
    "TextClassifier",
]
