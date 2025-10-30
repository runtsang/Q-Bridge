"""IntegratedQuanvolutionTransformer – classical backbone for image classification.

This module fuses a quantum‑inspired convolutional front‑end (QuanvolutionFilter)
with a transformer backbone that uses only classical attention and feed‑forward
blocks.  The model can be trained end‑to‑end with PyTorch and is fully
compatible with the original Quanvolution and Transformer examples.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1.  Quanvolution block – depth‑wise separable, fully trainable
# --------------------------------------------------------------------------- #

class QuanvolutionFilter(nn.Module):
    """Depth‑wise separable convolution that mimics a 2×2 quantum kernel.
    The outer conv layers are fully trainable, the inner 2×2 kernel is
    replaced by a learned 4‑channel stride‑2 filter that preserves the
    channel dimension of the original patch."""
    def __init__(self, in_channels: int = 1, depth_channels: int = 4, kernel_size: int = 2) -> None:
        super().__init__()
        # depth‑wise conv -> in_channels → depth_channels
        self.depthwise = nn.Conv2d(in_channels, depth_channels, kernel_size=kernel_size,
                                   stride=kernel_size, groups=in_channels)
        # point‑wise conv -> collapse back to in_channels for compatibility
        self.pointwise = nn.Conv2d(depth_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W), H=W=28 for MNIST
        x = self.depthwise(x)          # (B, depth_channels, H/2, W/2)
        x = self.pointwise(x)          # (B, C, H/2, W/2)
        return x.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    """Classical classifier that uses the separable filter followed by a linear head."""
    def __init__(self, in_channels: int = 1, depth_channels: int = 4, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels, depth_channels)
        self.linear = nn.Linear(depth_channels * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.qfilter(x)
        logits = self.linear(feats)
        return F.log_softmax(logits, dim=-1)

# --------------------------------------------------------------------------- #
# 2.  Transformer backbone – classical sub‑modules
# --------------------------------------------------------------------------- #

class MultiHeadAttentionBase(nn.Module):
    """Base class shared by classical and quantum attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        return x.view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        return self.dropout(scores)

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        k = self.separate_heads(k)
        q = self.separate_heads(q)
        v = self.separate_heads(v)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(x.size(0), -1, self.embed_dim)
        return self.out(out)

class FeedForwardBase(nn.Module):
    """Base for the feed‑forward sub‑module."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

class FeedForwardClassical(FeedForwardBase):
    """Two‑layer MLP."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface only
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
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

# --------------------------------------------------------------------------- #
# 3.  Integrated model – quanvolution + transformer
# --------------------------------------------------------------------------- #

class IntegratedQuanvolutionTransformer(nn.Module):
    """Full model that applies a quanvolution filter to an image, embeds the resulting patches,
    passes them through a classical transformer, and produces a classification score."""
    def __init__(self,
                 in_channels: int = 1,
                 depth_channels: int = 4,
                 embed_dim: int = 64,
                 num_heads: int = 8,
                 num_blocks: int = 4,
                 ffn_dim: int = 256,
                 num_classes: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels, depth_channels)
        self.embed_proj = nn.Linear(depth_channels, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
              for _ in range(num_blocks)]
        )
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # 1. Classical quanvolution
        feat = self.qfilter(x)                     # (B, depth_channels * 14 * 14)
        seq_len = 14 * 14
        feat = feat.view(x.size(0), seq_len, -1)   # (B, seq_len, depth_channels)
        # 2. Project to embedding dimension
        feat = self.embed_proj(feat)               # (B, seq_len, embed_dim)
        # 3. Positional encoding
        feat = self.pos_encoder(feat)              # (B, seq_len, embed_dim)
        # 4. Classical transformer blocks
        out = self.transformer(feat)               # (B, seq_len, embed_dim)
        # 5. Pooling and classification
        out = out.mean(dim=1)                      # (B, embed_dim)
        out = self.dropout(out)
        return self.classifier(out)

__all__ = [
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "IntegratedQuanvolutionTransformer",
]
