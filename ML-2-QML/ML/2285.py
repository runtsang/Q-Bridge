"""Hybrid classical transformer classifier for binary classification.

This module implements a CNN backbone followed by a stack of classical transformer
blocks and a linear classification head.  It is a drop‑in replacement for the
original `QCNet` but fully classical, making it suitable for environments
without quantum back‑ends.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HybridQuantumTransformerClassifier"]


class ConvFeatureExtractor(nn.Module):
    """Small CNN to extract image features."""
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            nn.Dropout2d(0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            nn.Dropout2d(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
    """Standard transformer block (self‑attention + FFN)."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(src, src, src)
        src = self.norm1(src + self.dropout(attn_out))
        ffn_out = self.ffn(src)
        return self.norm2(src + self.dropout(ffn_out))


class HybridQuantumTransformerClassifier(nn.Module):
    """CNN + transformer + linear head for binary classification."""
    def __init__(self,
                 embed_dim: int = 128,
                 n_heads: int = 4,
                 n_blocks: int = 2,
                 d_ff: int = 256,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = ConvFeatureExtractor()
        self.flatten = nn.Flatten(start_dim=1)
        self.proj = nn.Linear(55815, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, n_heads, d_ff, dropout) for _ in range(n_blocks)]
        )
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.proj(x).unsqueeze(1)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        probs = torch.sigmoid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)
