"""Classical transformer with optional kernel‑based attention and fast evaluation utilities."""

from __future__ import annotations

import math
from typing import Optional, Iterable, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---------- Core building blocks ----------
class MultiHeadAttentionBase(nn.Module):
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
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        q = self.q_linear(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out(out)

class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# ---------- Kernel‑based attention ----------
class KernelAttention(MultiHeadAttentionBase):
    """Attention that uses an RBF kernel to compute similarity."""
    def __init__(self, embed_dim: int, num_heads: int, gamma: float = 1.0, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.gamma = gamma

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        diff = x.unsqueeze(2) - x.unsqueeze(1)  # (b, s, s, d)
        dist2 = (diff ** 2).sum(-1)
        K = torch.exp(-self.gamma * dist2)
        if mask is not None:
            K = K.masked_fill(mask.unsqueeze(1) == 0, 0.0)
        attn = F.softmax(K, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, x)

class TransformerBlockKernel(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, gamma: float, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = KernelAttention(embed_dim, num_heads, gamma, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# ---------- Positional encoding ----------
class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# ---------- Fast estimator ----------
class FastEstimator:
    """Simple evaluator that runs a model on a list of inputs and returns logits."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(self, inputs: Iterable[torch.Tensor]) -> List[torch.Tensor]:
        self.model.eval()
        results = []
        with torch.no_grad():
            for inp in inputs:
                results.append(self.model(inp))
        return results

# ---------- Hybrid transformer ----------
class QTransformer(nn.Module):
    """
    Hybrid transformer that can operate in a fully classical mode or replace the
    attention sub‑module with a kernel‑based variant. The API is identical to
    the classical version, enabling side‑by‑side experiments.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_kernel_attention: bool = False,
                 kernel_gamma: float = 1.0) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        if use_kernel_attention:
            self.transformer = nn.Sequential(
                *[TransformerBlockKernel(embed_dim, num_heads, ffn_dim, kernel_gamma, dropout)
                  for _ in range(num_blocks)]
            )
        else:
            self.transformer = nn.Sequential(
                *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                  for _ in range(num_blocks)]
            )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockKernel",
    "KernelAttention",
    "PositionalEncoder",
    "FastEstimator",
    "QTransformer",
]
