import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionBase(nn.Module):
    """
    Base class for multi‑head attention.  Mirrors the original seed but
    keeps the API generic so that subclasses may replace the linear
    projections with a quantum module.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """
    Classic multi‑head attention implementation using torch.nn.Linear.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_heads, -1).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim // self.num_heads)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        attn = torch.matmul(probs, v).transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_proj(attn)

class FeedForwardBase(nn.Module):
    """
    Base class for the feed‑forward sub‑network.
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
    Classic two‑layer feed‑forward network.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class QuantumFeedForward(FeedForwardBase):
    """
    A toy quantum‑inspired feed‑forward block implemented purely with
    torch operations.  It simulates a small variational circuit that
    applies a parameterised rotation to each feature channel.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        # simulated quantum parameters
        self.theta = nn.Parameter(torch.randn(ffn_dim, 2))
        self.mu = nn.Parameter(torch.randn(ffn_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear1(x)
        # simulate quantum rotation: cos(theta * (z + mu))
        z = torch.cos(z.unsqueeze(-1) * self.theta + self.mu)
        z = z.squeeze(-1)
        return self.dropout(z)

class HybridTransformerBlock(nn.Module):
    """
    Transformer block that can optionally replace the feed‑forward part
    with a quantum‑inspired feed‑forward.  The flag `use_quantum_ffn`
    controls which sub‑network is used.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, use_quantum_ffn: bool = False):
        super().__init__()
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, dropout) if use_quantum_ffn \
                   else FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    """
    Sinusoidal positional encoding used by the original seed.
    """
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class QuantumHybridTransformer(nn.Module):
    """
    Main text classifier.  It mirrors the interface of the original seed
    but accepts an additional keyword `use_quantum` that toggles the
    quantum‑inspired feed‑forward within each transformer block.
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
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[HybridTransformerBlock(embed_dim, num_heads, ffn_dim,
                                     dropout, use_quantum_ffn=use_quantum)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "QuantumFeedForward",
    "HybridTransformerBlock",
    "PositionalEncoder",
    "QuantumHybridTransformer",
]
