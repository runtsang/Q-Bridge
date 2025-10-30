"""Hybrid transformer that merges classical, kernel, and quantum-inspired ideas."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------------------------------------------------------------------------- #
# Kernel utilities
# --------------------------------------------------------------------------- #
class RBFKernel(nn.Module):
    """Radial basis function kernel with trainable gamma."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: (batch, seq, dim)
        diff = x.unsqueeze(2) - y.unsqueeze(1)  # (batch, seq_x, seq_y, dim)
        dist_sq = (diff * diff).sum(dim=-1)  # (batch, seq_x, seq_y)
        return torch.exp(-self.gamma * dist_sq)

# --------------------------------------------------------------------------- #
# Attention modules
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, seq, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch, seq, heads * d_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = self.combine_heads(out)
        return self.out_proj(out)

class KernelAttention(MultiHeadAttentionBase):
    """Attention that uses a kernel similarity matrix instead of dot‑product."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, gamma: float = 1.0) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.kernel = RBFKernel(gamma)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # compute kernel similarity between tokens
        kernel_matrix = self.kernel(x, x)  # (batch, seq, seq)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            kernel_matrix = kernel_matrix.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(kernel_matrix, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, x)

# --------------------------------------------------------------------------- #
# Feed‑forward modules
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
# Transformer block
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, use_kernel: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = KernelAttention(embed_dim, num_heads, dropout) if use_kernel else MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# Positional encoding
# --------------------------------------------------------------------------- #
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
        return x + self.pe[:, :x.size(1)]

# --------------------------------------------------------------------------- #
# Image feature extractor (NAT style)
# --------------------------------------------------------------------------- #
class ConvFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)

# --------------------------------------------------------------------------- #
# Hybrid transformer
# --------------------------------------------------------------------------- #
class HybridTransformer(nn.Module):
    """
    A transformer that can operate in several modes:
    * classic attention + feed‑forward (default)
    * kernel‑based attention
    * image‑based feature extraction (NAT style)
    * classification or regression head
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
                 task: str = "classification",
                 use_image: bool = False) -> None:
        super().__init__()
        self.task = task
        self.use_image = use_image
        self.token_embedding = nn.Embedding(vocab_size, embed_dim) if not use_image else None
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlockClassical(embed_dim, num_heads, ffn_dim,
                                       dropout=dropout, use_kernel=use_kernel_attention)
             for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        if task == "regression":
            self.head = nn.Linear(embed_dim, 1)
        else:
            self.head = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        if use_image:
            self.image_extractor = ConvFeatureExtractor()
            self.image_proj = nn.Linear(4, embed_dim)
        else:
            self.image_extractor = None
            self.image_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_image:
            img_feat = self.image_extractor(x)
            x = self.image_proj(img_feat)
        else:
            tokens = self.token_embedding(x)
            x = self.pos_encoder(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x.mean(dim=1))
        return self.head(x)

__all__ = ["HybridTransformer"]
