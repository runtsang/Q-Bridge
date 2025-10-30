import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionBase(nn.Module):
    """Base class providing utilities for head splitting/combining."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        return x.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, seq, dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Purely classical multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, bias: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q, k, v = (self._split_heads(t) for t in (q, k, v))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask[:, None, :, :], -1e9)
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        attn = torch.matmul(probs, v)
        return self._combine_heads(attn)

class FeedForwardBase(nn.Module):
    """Base class for feed‑forward sub‑modules."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

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
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

class TransformerBlockClassical(TransformerBlockBase):
    """Classical transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

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
        return x + self.pe[:, :x.size(1)]

class HybridTextClassifier(nn.Module):
    """Drop‑in replacement for TextClassifier that can optionally enable quantum sub‑modules."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
        n_qlayers: int = 1,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoder(embed_dim)
        # In the classical implementation we ignore quantum flags and always use classical blocks
        self.blocks = nn.ModuleList(
            [TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)
