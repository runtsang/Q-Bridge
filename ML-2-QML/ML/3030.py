"""Purely classical hybrid model combining a CNN, linear projection, and transformer classifier."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------- CNN Feature Extractor ------------------- #
class ResCNN(nn.Module):
    """Simple CNN with residual connection."""
    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.res = nn.Conv2d(base_channels * 2, base_channels * 2, 1)
        self.norm = nn.BatchNorm2d(base_channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return self.norm(x2 + self.res(x2))

# ------------------- Linear Projection ------------------- #
class LinearProjection(nn.Module):
    """Projection from flattened CNN features to embedding dimension."""
    def __init__(self, in_features: int, embed_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

# ------------------- Positional Encoding ------------------- #
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
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

# ------------------- Transformer Blocks ------------------- #
class MultiHeadAttention(nn.Module):
    """Classical multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        qkv = self.qkv(x).reshape(batch, seq, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.out_proj(out)

class FeedForward(nn.Module):
    """Classical two‑layer feed‑forward."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Transformer block with residual connections."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)

# ------------------- Text Classifier ------------------- #
class TextClassifier(nn.Module):
    """Transformer‑based classifier."""
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_blocks: int,
                 ffn_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(x)
        x = self.pos_emb(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

# ------------------- Hybrid Model ------------------- #
class QuantumNATHybrid(nn.Module):
    """Hybrid model: CNN → linear projection → transformer classifier."""
    def __init__(self,
                 in_channels: int = 1,
                 base_channels: int = 16,
                 vocab_size: int = 30522,
                 embed_dim: int = 64,
                 num_heads: int = 4,
                 num_blocks: int = 4,
                 ffn_dim: int = 128,
                 num_classes: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.cnn = ResCNN(in_channels, base_channels)
        self.proj = LinearProjection(base_channels * 2 * 7 * 7, embed_dim)
        self.classifier = TextClassifier(vocab_size, embed_dim, num_heads,
                                         num_blocks, ffn_dim, num_classes, dropout)

    def forward(self, img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        # img: (B, C, H, W), txt: (B, L)
        feat = self.cnn(img)
        feat = feat.view(feat.size(0), -1)
        feat = self.proj(feat)
        # embed image features as a single token
        img_token = feat.unsqueeze(1)  # (B, 1, E)
        txt_token = self.classifier.token_emb(txt)  # (B, L, E)
        tokens = torch.cat([img_token, txt_token], dim=1)
        tokens = self.classifier.pos_emb(tokens)
        x = self.classifier.transformer(tokens)
        x = x.mean(dim=1)
        x = self.classifier.dropout(x)
        return self.classifier.classifier(x)

__all__ = ["QuantumNATHybrid"]
