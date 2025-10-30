"""Hybrid transformer that supports optional quantum attention and contrastive learning."""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Core building blocks – classical + quantum wrappers
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Base class used by all attention variants."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split last dim into (num_heads, head_dim)."""
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented classically."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.embed_dim)
        return self.out_proj(out)

class MultiHeadAttentionQuantumWrapper(MultiHeadAttentionBase):
    """
    Wrapper that delegates the projection of Q, K, V to an external callable.
    The callable should accept a tensor of shape (B, L, D) and return a tuple
    (Q, K, V) each of shape (B, L, D). If None, classical linear layers are used.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        projection_func: Optional[Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.projection_func = projection_func
        if projection_func is None:
            self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
            self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
            self.wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.projection_func is None:
            q = self.wq(x)
            k = self.wk(x)
            v = self.wv(x)
        else:
            q, k, v = self.projection_func(x)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.embed_dim)
        return self.out_proj(out)

# --------------------------------------------------------------------------- #
# 2. Feed‑forward networks
# --------------------------------------------------------------------------- #
class FeedForwardBase(nn.Module):
    """Base class for feed‑forward layers."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    """Two‑layer perceptron feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# --------------------------------------------------------------------------- #
# 3. Transformer blocks
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class TransformerBlockHybrid(TransformerBlockBase):
    """
    Hybrid transformer block that can switch between classical and quantum
    attention/FFN components. Weight sharing across layers is handled by the
    outer classifier module.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_quantum: bool = False,
        projection_func: Optional[Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        if use_quantum:
            self.attn = MultiHeadAttentionQuantumWrapper(
                embed_dim, num_heads, dropout, projection_func=projection_func
            )
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        else:
            self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# 4. Positional encoding
# --------------------------------------------------------------------------- #
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

# --------------------------------------------------------------------------- #
# 5. Contrastive projection head
# --------------------------------------------------------------------------- #
class ContrastiveHead(nn.Module):
    """Simple projection head for contrastive learning."""
    def __init__(self, embed_dim: int, projection_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --------------------------------------------------------------------------- #
# 6. Hybrid text classifier
# --------------------------------------------------------------------------- #
class TextClassifierHybrid(nn.Module):
    """
    Transformer‑based text classifier that can mix classical and quantum
    sub‑modules, optionally share weights across blocks, and provide a
    contrastive projection head.
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
        share_weights: bool = False,
        use_quantum: bool = False,
        projection_func: Optional[Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
        contrastive_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        block_cls = (
            lambda: TransformerBlockHybrid(
                embed_dim, num_heads, ffn_dim, dropout, use_quantum, projection_func
            )
        )
        if share_weights:
            single_block = block_cls()
            blocks = nn.ModuleList([single_block] * num_blocks)
        else:
            blocks = nn.ModuleList([block_cls() for _ in range(num_blocks)])
        self.transformer = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        self.contrastive_head = (
            ContrastiveHead(embed_dim, contrastive_dim) if contrastive_dim is not None else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns logits. If a contrastive head is present, also returns the
        projected embeddings.
        """
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        logits = self.classifier(x)
        if self.contrastive_head is None:
            return logits
        return logits, self.contrastive_head(x)

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionQuantumWrapper",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockHybrid",
    "PositionalEncoder",
    "ContrastiveHead",
    "TextClassifierHybrid",
]
