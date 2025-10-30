"""Transformer‑based sampler network with a hybrid sigmoid head.

The network embeds a 2‑dimensional input into a higher‑dimensional space,
passes it through a stack of multi‑head attention and feed‑forward layers,
and finally projects to a 2‑class probability distribution.  A lightweight
Hybrid head mimics the behaviour of a quantum expectation layer, making
the architecture a natural drop‑in replacement for the quantum version
in downstream pipelines.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding used by the transformer."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class TransformerBlockClassical(nn.Module):
    """A standard transformer block with multi‑head self‑attention."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class Hybrid(nn.Module):
    """Differentiable sigmoid head that emulates a quantum expectation."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return torch.sigmoid(self.linear(logits) + self.shift)

class SamplerQNNGen150(nn.Module):
    """Transformer‑based sampler producing a 2‑class probability distribution."""
    def __init__(self,
                 embed_dim: int = 16,
                 num_heads: int = 4,
                 ffn_dim: int = 64,
                 num_blocks: int = 2,
                 dropout: float = 0.1,
                 shift: float = 0.0):
        super().__init__()
        self.embedding = nn.Linear(2, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim,
                                       dropout) for _ in range(num_blocks)]
        )
        self.classifier = nn.Linear(embed_dim, 2)
        self.hybrid = Hybrid(2, shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # aggregate across sequence
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=-1)
        return self.hybrid(probs)

__all__ = ["SamplerQNNGen150"]
