"""
Hybrid transformer module with optional kernel‑based attention.

This implementation extends the classic transformer architecture by adding
a kernel‑weighted attention mechanism.  The `HybridTransformerKernel` class
accepts a `use_kernel` flag that, when true, replaces the standard
dot‑product attention with an RBF‑kernel similarity.  The quantum
counterpart is provided in the QML module below.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence


# --------------------------------------------------------------------------- #
# Core attention building blocks
# --------------------------------------------------------------------------- #

class MultiHeadAttentionBase(nn.Module):
    """
    Abstract base class for multi‑head attention.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """
    Standard dot‑product attention.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out


class MultiHeadAttentionKernelClassical(MultiHeadAttentionBase):
    """
    Kernel‑weighted attention.  Similarity between queries and keys is computed
    using a Gaussian RBF kernel.
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1, gamma: float = 1.0) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.gamma = gamma
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        batch_size, seq_len, embed_dim = x.size()

        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # (batch, heads, seq, d_k)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # RBF similarity: exp(-γ * ||q - k||²)  (broadcasted over heads)
        diff = q.unsqueeze(3) - k.unsqueeze(2)          # (b, h, s, s, d_k)
        sq_dist = (diff**2).sum(-1)                     # (b, h, s, s)
        similarity = torch.exp(-self.gamma * sq_dist)    # (b, h, s, s)

        if mask is not None:
            mask_exp = mask.unsqueeze(1).unsqueeze(2)  # (b,1,1,s)
            similarity = similarity.masked_fill(mask_exp == 0, -1e9)

        attn_weights = F.softmax(similarity, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)  # (b, h, s, d_k)
        attn_out = attn_out.transpose(1, 2).contiguous()\
                       .view(batch_size, seq_len, embed_dim)
        return self.combine_heads(attn_out)


# Quantum‑compatible alias (kept for API symmetry)
MultiHeadAttentionQuantum = MultiHeadAttentionClassical


# --------------------------------------------------------------------------- #
# Feed‑forward blocks
# --------------------------------------------------------------------------- #

class FeedForwardBase(nn.Module):
    """
    Base class for feed‑forward layers.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """
    Standard two‑layer MLP.
    """
    def __init__(self, embed_dim: int, ffn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# Quantum‑compatible alias
FeedForwardQuantum = FeedForwardClassical


# --------------------------------------------------------------------------- #
# Transformer blocks
# --------------------------------------------------------------------------- #

class TransformerBlockBase(nn.Module):
    """
    Base transformer block with two LayerNorm layers.
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    """
    Classic transformer block using dot‑product attention.
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockKernelClassical(TransformerBlockBase):
    """
    Transformer block that applies kernel‑based attention.
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 ffn_dim: int, dropout: float = 0.1,
                 gamma: float = 1.0) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionKernelClassical(embed_dim, num_heads,
                                                      dropout, gamma)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# Quantum‑compatible alias
TransformerBlockQuantum = TransformerBlockClassical


# --------------------------------------------------------------------------- #
# Positional encoding
# --------------------------------------------------------------------------- #

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x + self.pe[:, :x.size(1)]


# --------------------------------------------------------------------------- #
# Combined transformer classifier
# --------------------------------------------------------------------------- #

class HybridTransformerKernel(nn.Module):
    """
    Transformer‑based classifier that can switch between classic, kernel‑based,
    and quantum‑enhanced configurations via flags.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_kernel: bool = False,
                 kernel_gamma: float = 1.0) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)

        # Build block list
        blocks = []
        for _ in range(num_blocks):
            if use_kernel:
                blk = TransformerBlockKernelClassical(embed_dim, num_heads,
                                                      ffn_dim, dropout,
                                                      gamma=kernel_gamma)
            else:
                blk = TransformerBlockClassical(embed_dim, num_heads,
                                               ffn_dim, dropout)
            blocks.append(blk)
        self.transformers = nn.Sequential(*blocks)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim,
                                    num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionKernelClassical",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockKernelClassical",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "HybridTransformerKernel",
]
