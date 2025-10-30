"""Hybrid transformer implementation with a classical‑quantum attention blend."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridAttentionBase(nn.Module):
    """Base class for hybrid attention that supports a mix between classical and quantum paths."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        mix_ratio: float = 1.0,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.mix_ratio = mix_ratio
        self.attn_weights: Optional[torch.Tensor] = None
        self.use_bias = use_bias

    def separate_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape and transpose for multi‑head attention."""
        batch_size = tensor.size(0)
        return tensor.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def compute_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Standard dot‑product attention."""
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class HybridAttentionClassical(HybridAttentionBase):
    """Purely classical attention – identical to the original."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_bias: bool = False,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout, mix_ratio=1.0, use_bias=use_bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        if embed_dim!= self.embed_dim:
            raise ValueError(f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})")
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        k = self.separate_heads(k)
        q = self.separate_heads(q)
        v = self.separate_heads(v)
        x = self.compute_attention(q, k, v)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.combine_heads(x)


class HybridAttentionQuantum(HybridAttentionClassical):
    """Hybrid attention that mixes classical and a simple quantum‑like transform."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        mix_ratio: float = 0.5,
        use_bias: bool = False,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.mix_ratio = mix_ratio
        # Simple quantum‑like linear transform to emulate quantum behaviour
        self.quantum_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        classical_out = super().forward(x, mask)
        quantum_out = self.quantum_proj(x)
        return self.mix_ratio * classical_out + (1.0 - self.mix_ratio) * quantum_out


class HybridFeedForwardBase(nn.Module):
    """Base class for feed‑forward blocks."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class HybridFeedForwardClassical(HybridFeedForwardBase):
    """Standard two‑layer feed‑forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class HybridFeedForwardQuantum(HybridFeedForwardClassical):
    """Alias for the classical feed‑forward in the ML version."""
    pass


class HybridTransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class HybridTransformerBlockClassical(HybridTransformerBlockBase):
    """Purely classical transformer block."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = HybridAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = HybridFeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class HybridTransformerBlockQuantum(HybridTransformerBlockBase):
    """Transformer block that optionally mixes classical and quantum attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        mix_ratio: float = 0.5,
        use_quantum_ffn: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = HybridAttentionQuantum(embed_dim, num_heads, dropout, mix_ratio)
        if use_quantum_ffn:
            self.ffn = HybridFeedForwardQuantum(embed_dim, ffn_dim, dropout)
        else:
            self.ffn = HybridFeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

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


class TextClassifierHybrid(nn.Module):
    """Transformer‑based text classifier supporting hybrid quantum‑classical layers."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        mix_ratio: float = 0.5,
        use_quantum_ffn: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoding(embed_dim)
        blocks = [
            HybridTransformerBlockQuantum(
                embed_dim,
                num_heads,
                ffn_dim,
                mix_ratio=mix_ratio,
                use_quantum_ffn=use_quantum_ffn,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ]
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        if num_classes > 2:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "HybridAttentionBase",
    "HybridAttentionClassical",
    "HybridAttentionQuantum",
    "HybridFeedForwardBase",
    "HybridFeedForwardClassical",
    "HybridFeedForwardQuantum",
    "HybridTransformerBlockBase",
    "HybridTransformerBlockClassical",
    "HybridTransformerBlockQuantum",
    "PositionalEncoding",
    "TextClassifierHybrid",
]
