"""Unified hybrid classifier with classical head.

This module defines `UnifiedHybridClassifier`, which can operate in three
modes: fully‑connected (`fc`), convolutional (`cnn`), or transformer
(`transformer`).  The implementation is entirely classical and relies on
PyTorch and NumPy.  It serves as a drop‑in replacement for the original
`FCL.py` while providing a richer feature extractor and a flexible
classification head that can be swapped for a quantum implementation in
the QML counterpart.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# 1.  Fully‑connected layer (from FCL.py)
# ------------------------------------------------------------------
class FullyConnectedLayer(nn.Module):
    """Simple fully‑connected layer with tanh activation."""

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        values = thetas.view(-1, 1).float()
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation

# ------------------------------------------------------------------
# 2.  Feature extractor – CNN
# ------------------------------------------------------------------
class _CNNFeatureExtractor(nn.Module):
    """Convolutional backbone that can be swapped into the hybrid classifier."""

    def __init__(self, out_features: int = 120) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(0.2)
        self.drop2 = nn.Dropout2d(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(55815, out_features)
        self.fc2 = nn.Linear(out_features, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ------------------------------------------------------------------
# 3.  Hybrid head – classical analogue
# ------------------------------------------------------------------
class _HybridHead(nn.Module):
    """Classical head that emulates a quantum expectation layer."""

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.linear(inputs)
        return torch.sigmoid(logits + self.shift)

# ------------------------------------------------------------------
# 4.  Transformer components – classical
# ------------------------------------------------------------------
class MultiHeadAttentionBase(nn.Module):
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
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output

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
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

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

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

# ------------------------------------------------------------------
# 5.  UnifiedHybridClassifier
# ------------------------------------------------------------------
class UnifiedHybridClassifier(nn.Module):
    """A versatile classifier that can operate in three modes:

    * ``mode='fc'`` – a simple fully‑connected layer (FCL).
    * ``mode='cnn'`` – a CNN backbone followed by a hybrid head.
    * ``mode='transformer'`` – a transformer encoder followed by a linear head.
    """

    def __init__(
        self,
        mode: str = "fc",
        *,
        n_features: int = 1,
        out_features: int = 120,
        num_heads: int = 4,
        ffn_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in {"fc", "cnn", "transformer"}:
            raise ValueError("mode must be one of 'fc', 'cnn', or 'transformer'")
        self.mode = mode
        self.num_classes = num_classes

        if mode == "fc":
            self.core = FullyConnectedLayer(n_features)
        elif mode == "cnn":
            self.extractor = _CNNFeatureExtractor(out_features)
            self.head = _HybridHead(out_features, shift=shift)
        else:  # transformer
            self.pos_encoder = PositionalEncoder(embed_dim=num_heads * ffn_dim)
            self.transformer = nn.Sequential(
                *[
                    TransformerBlockClassical(num_heads * ffn_dim, num_heads, ffn_dim)
                    for _ in range(num_layers)
                ]
            )
            self.fc = nn.Linear(num_heads * ffn_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "fc":
            return self.core(x)
        elif self.mode == "cnn":
            feats = self.extractor(x)
            probs = self.head(feats)
            if self.num_classes == 2:
                return torch.cat([probs, 1 - probs], dim=-1)
            return probs
        else:  # transformer
            x = self.pos_encoder(x)
            x = self.transformer(x)
            x = x.mean(dim=1)  # global average pooling
            logits = self.fc(x)
            if self.num_classes == 2:
                probs = torch.sigmoid(logits)
                return torch.cat([probs, 1 - probs], dim=-1)
            return logits

__all__ = [
    "FullyConnectedLayer",
    "UnifiedHybridClassifier",
    "PositionalEncoder",
    "TransformerBlockClassical",
    "FeedForwardClassical",
    "MultiHeadAttentionClassical",
]
