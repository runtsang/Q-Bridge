"""Hybrid transformer with optional quantum sub‑modules for ablation experiments."""

from __future__ import annotations

import math
from typing import Optional, Iterable, Tuple, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionBase(nn.Module):
    """Base class for all attention variants."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape to (batch, heads, seq, d_k) for attention."""
        batch_size = x.size(0)
        return (
            x.view(batch_size, -1, self.num_heads, self.d_k)
           .transpose(1, 2)
           .contiguous()
        )

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard scaled‑dot‑product attention with optional mask."""
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

    def forward(self, *args, **kwargs) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with PyTorch."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        q, k, v = self.separate_heads(q), self.separate_heads(k), self.separate_heads(v)
        attn, _ = self.attention(q, k, v, mask)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_proj(attn)


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward networks."""

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


class TransformerBlockBase(nn.Module):
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


class TransformerBlockClassical(TransformerBlockBase):
    """Purely classical transformer block."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockHybrid(TransformerBlockBase):
    """
    Hybrid transformer block that can swap in quantum sub‑modules at runtime.
    The quantum modules must implement the same interface as the classical ones.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        *,
        attn_module: Optional[Callable[[], nn.Module]] = None,
        ffn_module: Optional[Callable[[], nn.Module]] = None,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        # Default to classical modules if none provided
        self.attn = attn_module() if attn_module else MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = ffn_module() if ffn_module else FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

    def replace_modules(
        self,
        attn_module: Optional[Callable[[], nn.Module]] = None,
        ffn_module: Optional[Callable[[], nn.Module]] = None,
    ) -> None:
        """Replace the attention or feed‑forward sub‑modules on the fly."""
        if attn_module:
            self.attn = attn_module()
        if ffn_module:
            self.ffn = ffn_module()


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TextClassifierHybrid(nn.Module):
    """
    Transformer‑based text classifier that supports optional quantum sub‑modules.
    The `quantum_mask` list determines which blocks use quantum layers.
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
        quantum_mask: Optional[List[bool]] = None,
        attn_factory: Optional[Callable[[], nn.Module]] = None,
        ffn_factory: Optional[Callable[[], nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.dropout = nn.Dropout(dropout)

        if quantum_mask is None:
            quantum_mask = [False] * num_blocks

        blocks: List[nn.Module] = []
        for use_q in quantum_mask:
            if use_q:
                block = TransformerBlockHybrid(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    dropout,
                    attn_module=attn_factory,
                    ffn_module=ffn_factory,
                )
            else:
                block = TransformerBlockClassical(
                    embed_dim, num_heads, ffn_dim, dropout
                )
            blocks.append(block)

        self.transformers = nn.Sequential(*blocks)

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

    def enable_quantum(self, block_indices: Iterable[int]) -> None:
        """
        Replace specified blocks with quantum variants.
        The quantum modules must be set via `attn_factory` and `ffn_factory`
        when constructing the model or by passing a callable that returns
        a new instance of the desired quantum module.
        """
        for idx in block_indices:
            block = self.transformers[idx]
            if isinstance(block, TransformerBlockHybrid):
                block.replace_modules(
                    attn_module=block.attn.__class__,
                    ffn_module=block.ffn.__class__,
                )
            else:
                raise TypeError(f"Block {idx} is not a hybrid block")

    def disable_quantum(self, block_indices: Iterable[int]) -> None:
        """Revert specified blocks back to classical implementations."""
        for idx in block_indices:
            block = self.transformers[idx]
            if isinstance(block, TransformerBlockHybrid):
                block.replace_modules(
                    attn_module=MultiHeadAttentionClassical,
                    ffn_module=FeedForwardClassical,
                )
            else:
                raise TypeError(f"Block {idx} is not a hybrid block")


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockHybrid",
    "PositionalEncoder",
    "TextClassifierHybrid",
]
