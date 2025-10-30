"""Hybrid transformer‑based text classifier with optional quantum layers.

This module provides a drop‑in replacement for the original TextClassifier
while adding rotary positional encoding, a learnable output projection,
and an `evaluate` helper for quick metrics.  The `use_quantum` flag
currently routes to the classical implementation but is kept for API
compatibility with the quantum version.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Attention primitives ----------
class MultiHeadAttentionBase(nn.Module):
    """Base class for attention mechanisms."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_bias = use_bias

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Rearrange tensor to (batch, heads, seq_len, d_k)."""
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
        """Scaled dot‑product attention."""
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with linear layers."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_bias: bool = False,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        attn_output, _ = self.attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            x.size(0), -1, self.embed_dim
        )
        return self.combine_heads(attn_output)


# ---------- Feed‑forward primitives ----------
class FeedForwardBase(nn.Module):
    """Base class for feed‑forward layers."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer MLP."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ---------- Transformer block ----------
class TransformerBlockBase(nn.Module):
    """Base transformer block."""

    def __init__(self, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)


class TransformerBlockClassical(TransformerBlockBase):
    """Classic transformer block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# ---------- Rotary positional encoding ----------
class RotaryPositionalEncoding(nn.Module):
    """Sinusoidal rotary positional encoding with learnable frequency."""

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        # learnable frequency parameter
        self.freq = nn.Parameter(torch.ones(embed_dim // 2) * 10000.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        position = torch.arange(seq_len, device=x.device).unsqueeze(1)
        inv_freq = 1.0 / (self.freq ** (torch.arange(0, self.embed_dim, 2, device=x.device) / self.embed_dim))
        sinusoid_inp = torch.einsum("i, j -> i j", position, inv_freq)
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        x_even = x[:, :, ::2]
        x_odd = x[:, :, 1::2]
        x_rotated = torch.cat([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1)
        return x_rotated


# ---------- Hybrid classifier ----------
class TextClassifierHybrid(nn.Module):
    """Hybrid transformer‑based text classifier.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    embed_dim : int
        Embedding dimension.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Hidden dimension of the feed‑forward layers.
    num_classes : int
        Number of output classes.
    dropout : float, optional
        Drop‑out probability.
    use_quantum : bool, optional
        Flag to enable quantum sub‑modules.  In the classical
        implementation this flag is ignored and the model
        behaves purely classically.
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
        use_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = RotaryPositionalEncoding(embed_dim)
        block_cls = TransformerBlockClassical  # classical block
        self.transformer_blocks = nn.ModuleList(
            [
                block_cls(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)          # (batch, seq_len, embed_dim)
        x = self.pos_encoder(x)              # rotary encoding
        for block in self.transformer_blocks:
            x = block(x)
        x = x.mean(dim=1)                    # global pooling
        x = self.dropout(self.output_proj(x))  # learnable projection
        return self.classifier(x)

    def evaluate(self, dataloader, loss_fn, device: torch.device = torch.device("cpu")) -> dict:
        """Simple evaluation loop returning loss and accuracy."""
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                logits = self.forward(inputs)
                loss = loss_fn(logits, targets)
                total_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(logits, dim=1) if logits.dim() > 1 else logits.squeeze()
                correct += (preds == targets).sum().item()
                total += inputs.size(0)
        return {"loss": total_loss / total, "accuracy": correct / total}


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "RotaryPositionalEncoding",
    "TextClassifierHybrid",
]
