import math
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionBase(nn.Module):
    """Base class for all attention variants with shared utilities."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split the last dimension into (num_heads, head_dim)."""
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Recombine heads into the original embedding shape."""
        batch, heads, seq, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = self.combine_heads(out)
        return self.out_proj(out)

class HybridMultiHeadAttention(nn.Module):
    """Attention that can operate in classical, quantum or hybrid mode."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        quantum_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        mode: str = "classical",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.quantum_fn = quantum_fn
        self.classical = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.classical(x, mask)
        if self.mode == "quantum" and self.quantum_fn is not None:
            out = self.quantum_fn(out)
        elif self.mode == "hybrid" and self.quantum_fn is not None:
            q_out = self.quantum_fn(out)
            out = 0.5 * (out + q_out)
        return out

class FeedForwardBase(nn.Module):
    """Base class for feed‑forward blocks."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    """Standard two‑layer feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class HybridFeedForward(nn.Module):
    """Feed‑forward that can use a quantum mapping."""
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        dropout: float = 0.1,
        quantum_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        mode: str = "classical",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.quantum_fn = quantum_fn
        self.classical = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.classical(x)
        if self.mode == "quantum" and self.quantum_fn is not None:
            out = self.quantum_fn(out)
        elif self.mode == "hybrid" and self.quantum_fn is not None:
            q_out = self.quantum_fn(out)
            out = 0.5 * (out + q_out)
        return out

class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""
    def __init__(self, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlock(TransformerBlockBase):
    """Hybrid transformer block using the hybrid attention and feed‑forward."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        quantum_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        mode: str = "classical",
    ) -> None:
        super().__init__(embed_dim, dropout)
        self.attn = HybridMultiHeadAttention(embed_dim, num_heads, dropout, quantum_fn, mode)
        self.ffn = HybridFeedForward(embed_dim, ffn_dim, dropout, quantum_fn, mode)

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
        return x + self.pe[:, : x.size(1)]

class TextClassifier(nn.Module):
    """Transformer‑based text classifier supporting hybrid attention and feed‑forward."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        mode: str = "classical",
        quantum_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                ffn_dim,
                dropout,
                quantum_fn,
                mode,
            )
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "HybridMultiHeadAttention",
    "FeedForwardBase",
    "FeedForwardClassical",
    "HybridFeedForward",
    "TransformerBlockBase",
    "TransformerBlock",
    "PositionalEncoder",
    "TextClassifier",
]
