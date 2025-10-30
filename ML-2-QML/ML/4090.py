"""
UnifiedQuantumHybridModel – Classical backbone for vision and text.

This module defines a single PyTorch `nn.Module` that can be instantiated either
as a vision classifier (image input) or a text classifier (token indices).
It merges ideas from:

*  Quantum‑NAT CNN‑FC (classical conv → FC)  – for image feature extraction.
*  QCNN quantum‑convolution + pooling – for a learnable feature projector.
*  Transformer blocks with optional quantum sub‑modules – for sequence processing.

The model is fully classical; only the architecture is inspired by the quantum
examples.  The interface is compatible with the anchor seed `QuantumNAT.py`,
allowing downstream pipelines to import `UnifiedQuantumHybridModel` in place
of `QFCModel`.  The code is self‑contained and ready for unit tests.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Shared utilities
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding used by the transformer head."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
# 2. Classical CNN backbone (from Quantum‑NAT)
# --------------------------------------------------------------------------- #
class _VisionCNN(nn.Module):
    """2‑D CNN followed by flatten‑to‑FC projection."""

    def __init__(self, in_channels: int = 1, out_features: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, out_features), nn.ReLU())
        self.norm = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.norm(x)


# --------------------------------------------------------------------------- #
# 3. QCNN‑style quantum convolution & pooling (classical emulation)
# --------------------------------------------------------------------------- #
class _QCNNProjector(nn.Module):
    """
    A lightweight, purely classical emulator of the QCNN quantum
    convolution + pooling architecture.  Each “convolution” step is a linear
    projection followed by a tanh activation – mirroring the quantum
    convolution kernels in the original QCNN paper.  Pooling reduces the
    feature dimension by a learned linear map.
    """

    def __init__(self, in_dim: int, proj_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
        )
        self.pool = nn.Sequential(
            nn.Linear(8, 12),
            nn.Tanh(),
            nn.Linear(12, 4),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.pool(x)
        return x


# --------------------------------------------------------------------------- #
# 4. Transformer block (classical + optional quantum sub‑modules)
# --------------------------------------------------------------------------- #
class _MultiHeadAttention(nn.Module):
    """Standard multi‑head attention with optional bias handling."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        q = q.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_proj(out)


class _FeedForward(nn.Module):
    """Two‑layer feed‑forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class _TransformerBlock(nn.Module):
    """Single transformer block with residuals and LayerNorm."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = _MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = _FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
# 5. Unified model
# --------------------------------------------------------------------------- #
class UnifiedQuantumHybridModel(nn.Module):
    """
    Multi‑modal hybrid architecture that can process either images or text.

    Parameters
    ----------
    modality : str
        ``"vision"`` or ``"text"``.  The default is ``"vision"``.
    in_channels : int, optional
        Number of image channels (default 1).  Ignored for text.
    vocab_size : int, optional
        Vocabulary size for text input.  Ignored for vision.
    embed_dim : int, default 128
        Dimensionality of the embedding / feature space.
    num_heads : int, default 4
        Number of attention heads.
    num_blocks : int, default 4
        Number of transformer blocks.
    ffn_dim : int, default 256
        Size of the feed‑forward network.
    num_classes : int, default 2
        Number of target classes.  If ``<=2`` a sigmoid output is used.
    """

    def __init__(
        self,
        modality: str = "vision",
        *,
        in_channels: int = 1,
        vocab_size: int = 0,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 4,
        ffn_dim: int = 256,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.modality = modality.lower()
        if self.modality not in {"vision", "text"}:
            raise ValueError("modality must be 'vision' or 'text'")

        # --- Vision branch --------------------------------------------------
        if self.modality == "vision":
            self.backbone = _VisionCNN(in_channels=in_channels, out_features=embed_dim)
            self.projector = _QCNNProjector(in_dim=embed_dim, proj_dim=embed_dim)
        else:  # text
            self.token_emb = nn.Embedding(vocab_size, embed_dim)
            self.pos_emb = PositionalEncoding(embed_dim)

        # --- Transformer backbone --------------------------------------------
        self.transformer = nn.Sequential(
            *[
                _TransformerBlock(embed_dim, num_heads, ffn_dim)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(0.1)

        # --- Classification head --------------------------------------------
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Vision path
        if self.modality == "vision":
            x = self.backbone(x)          # -> (B, E)
            x = self.projector(x)         # -> (B, E)
            x = x.unsqueeze(1)            # -> (B, 1, E)
        else:  # text
            x = self.token_emb(x)         # -> (B, T, E)
            x = self.pos_emb(x)           # -> (B, T, E)

        # Transformer
        x = self.transformer(x)          # -> (B, S, E)
        x = self.dropout(x.mean(dim=1))  # global pooling

        return self.classifier(x)
