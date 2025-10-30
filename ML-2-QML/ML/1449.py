# Hybrid transformer with parameter‑shared quantum attention and classical feed‑forward.
# The TextClassifier class supports both a fully classical configuration and a hybrid
# configuration where the attention heads use a shared linear projection that mimics a
# parameter‑shared quantum circuit.  The same class can be toggled to disable
# LayerNorm or to add a gated residual connection, enabling ablation studies
# without changing the public API.

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Base classes
class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out

# Parameter‑shared quantum attention – classical simulation
class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    # Classical simulation of a quantum‑style attention head with shared parameters.
    # The same linear layer is used for all Q, K, V projections, emulating a
    # parameter‑shared quantum circuit.
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.qkv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, embed_dim = x.size()
        head_dim = embed_dim // self.num_heads
        x = x.view(batch, seq_len, self.num_heads, head_dim).transpose(1, 2)
        return x

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        proj = self.qkv(x)
        q, k, v = proj.chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
        return self.out_proj(attn_output)

# Feed‑forward layers
class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class FeedForwardQuantum(FeedForwardBase):
    # Placeholder for a quantum feed‑forward block.  In the hybrid version
    # used by the QML module this will be replaced by a real quantum circuit.
    # The classical fallback simply re‑uses the classical MLP.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

# Positional encoding
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

# Transformer block
class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_layernorm: bool = True) -> None:
        super().__init__()
        self.use_layernorm = use_layernorm
        self.norm1 = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, use_layernorm: bool = True, gated_residual: bool = False):
        super().__init__(embed_dim, num_heads, dropout, use_layernorm)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.gated_residual = gated_residual
        if gated_residual:
            self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out = self.attn(x)
        if self.gated_residual:
            gate = torch.sigmoid(self.gate)
            x = self.norm1(x + gate * self.dropout(attn_out))
        else:
            x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, use_layernorm: bool = True, gated_residual: bool = False):
        super().__init__(embed_dim, num_heads, dropout, use_layernorm)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, dropout)
        self.gated_residual = gated_residual
        if gated_residual:
            self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out = self.attn(x)
        if self.gated_residual:
            gate = torch.sigmoid(self.gate)
            x = self.norm1(x + gate * self.dropout(attn_out))
        else:
            x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# Text classifier
class TextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        gated_residual: bool = False,
        quantum: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        block_cls = TransformerBlockQuantum if quantum else TransformerBlockClassical
        self.transformer_blocks = nn.ModuleList(
            [block_cls(embed_dim, num_heads, ffn_dim,
                       dropout=dropout,
                       use_layernorm=use_layernorm,
                       gated_residual=gated_residual)
             for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for blk in self.transformer_blocks:
            x = blk(x)
        x = self.dropout(x.mean(dim=1))
        return self.output(x)

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardQuantum",
    "PositionalEncoder",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockQuantum",
    "TextClassifier",
]
