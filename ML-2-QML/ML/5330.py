"""Hybrid quantum‑nat inspired classical model combining CNN, transformer, and classifier.

This module extends the original QuantumNAT by adding a stack of Transformer blocks
to capture long‑range dependencies.  The architecture remains fully classical
and can be used as a drop‑in replacement for the legacy QFCModel.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- CNN backbone ----
class _CNNBackbone(nn.Module):
    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ---- Positional encoding ----
class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

# ---- Transformer block (classical) ----
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# ---- Main hybrid model ----
class HybridQuantumNAT(nn.Module):
    """Classical hybrid model that fuses a CNN backbone with a stack of Transformer blocks."""
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 128,
        num_blocks: int = 2,
        num_classes: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = _CNNBackbone(in_channels)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)                  # [B, embed_dim]
        x = x.unsqueeze(1)                    # [B, 1, embed_dim] for transformer
        x = self.pos_encoder(x)
        x = self.transformers(x)
        x = x.mean(dim=1)                     # aggregate over sequence length
        x = self.dropout(x)
        return self.classifier(x)

__all__ = ["HybridQuantumNAT"]
