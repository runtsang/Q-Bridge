"""Hybrid CNN + Transformer classifier with optional quantum layers.

The module builds on the original QFCModel and QTransformerTorch by adding a
classical CNN backbone for feature extraction, a transformer stack for
sequence modeling, and a final classification head.  The design keeps the
same public API as the seed but enhances expressivity and
test‑ability.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalFeatureExtractor(nn.Module):
    """Four‑channel CNN that mirrors the original QFCModel."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        f = self.features(x)
        flat = f.view(bs, -1)
        out = self.fc(flat)
        return self.norm(out)

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding used by the transformer."""
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

class SimpleMultiHeadAttention(nn.Module):
    """Classical multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, seq_len, _ = x.size()
        q = self.q_proj(x).view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bs, seq_len, self.embed_dim)
        return self.out_proj(out)

class SimpleFeedForward(nn.Module):
    """Two‑layer feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class HybridTransformerBlock(nn.Module):
    """Transformer block that can be extended to quantum sub‑modules."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = SimpleMultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = SimpleFeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class QuantumHybridNat(nn.Module):
    """Hybrid CNN + Transformer classifier."""
    def __init__(
        self,
        vocab_size: int = 5000,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 128,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.feature_extractor = ClassicalFeatureExtractor()
        self.token_proj = nn.Linear(4, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[HybridTransformerBlock(embed_dim, num_heads, ffn_dim) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)          # (bs, 4)
        tokens = self.token_proj(feats).unsqueeze(1)  # (bs, 1, embed_dim)
        tokens = self.pos_encoder(tokens)
        tokens = self.transformers(tokens)
        pooled = tokens.mean(dim=1)
        out = self.dropout(pooled)
        return self.classifier(out)
