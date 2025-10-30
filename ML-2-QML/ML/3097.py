"""Hybrid quanvolution‑transformer architecture for image classification.

The module exposes two public classes:
  * `HybridQuanvolutionFilter` – a pure‑PyTorch implementation that replaces the 2×2 kernel with a stride‑2 conv
    followed by flattening.  It can be swapped with the quantum filter defined in the QML module.
  * `HybridQuanvolutionClassifier` – a classifier that chains the filter, a positional encoder,
    and a transformer stack.  The transformer can be instantiated with classical or quantum sub‑modules
    by passing the `quantum` flag.

This design keeps the classical API identical to the original seed while providing a seamless
path to quantum augmentation.  The code is fully importable and can be dropped into the
anchor path `Quanvolution.py` without breaking existing imports.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQuanvolutionFilter(nn.Module):
    """Flattened 2×2 patch extractor implemented purely in PyTorch."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)                 # (B, 4, 14, 14)
        return features.view(x.size(0), -1)     # (B, 4*14*14)

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding for 1‑D sequences."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x + self.pe[:, : x.size(1)]

class TransformerBlock(nn.Module):
    """Single transformer encoder layer with optional quantum sub‑modules."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        *,
        q_attn_layer: nn.Module | None = None,
        q_ffn_layer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Attention
        if q_attn_layer is not None:
            self.attn = q_attn_layer
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True
            )

        # Feed‑forward
        if q_ffn_layer is not None:
            self.ffn = q_ffn_layer
        else:
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.ReLU(),
                nn.Linear(ffn_dim, embed_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Attention sub‑layer
        if isinstance(self.attn, nn.MultiheadAttention):
            attn_out, _ = self.attn(x, x, x)
        else:
            attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed‑forward sub‑layer
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class HybridQuanvolutionClassifier(nn.Module):
    """End‑to‑end image classifier with optional quantum attention/FFN."""
    def __init__(
        self,
        num_classes: int,
        *,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        q_attn_layer: nn.Module | None = None,
        q_ffn_layer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.filter = HybridQuanvolutionFilter()
        self.pos_enc = PositionalEncoder(embed_dim)
        self.token_proj = nn.Linear(4, embed_dim)
        self.transformer = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    dropout,
                    q_attn_layer=q_attn_layer,
                    q_ffn_layer=q_ffn_layer,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        feats = self.filter(x)                           # (B, 4*14*14)
        seq = feats.view(x.size(0), -1, 4)                # (B, 196, 4)
        seq = self.token_proj(seq)                        # (B, 196, embed_dim)
        seq = self.pos_enc(seq)                           # (B, 196, embed_dim)
        seq = self.transformer(seq)                       # (B, 196, embed_dim)
        out = seq.mean(dim=1)                             # (B, embed_dim)
        out = self.dropout(out)
        return self.classifier(out)

__all__ = [
    "HybridQuanvolutionFilter",
    "HybridQuanvolutionClassifier",
]
