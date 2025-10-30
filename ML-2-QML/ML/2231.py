"""Hybrid transformer classifier – classical implementation.

The module mirrors the original QTransformerTorch API but adds:
* a convolutional feature extractor for image data (from QuantumNAT),
* a configurable front‑end (text or image),
* optional flags to enable quantum‑style blocks (ignored in the classical version).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

# --------------------------------------------------------------------------- #
# Core building blocks – classical
# --------------------------------------------------------------------------- #

class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention implemented purely with PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)


class FeedForwardClassical(nn.Module):
    """Two‑layer MLP used in transformer blocks."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlockClassical(nn.Module):
    """Standard transformer block with classical attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding compatible with the original API."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# --------------------------------------------------------------------------- #
# Convolutional encoder – inspired by QuantumNAT
# --------------------------------------------------------------------------- #

class ConvolutionalFeatureExtractor(nn.Module):
    """Simple 2‑D CNN that outputs a fixed‑size feature vector."""
    def __init__(self, output_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
        )
        self.norm = nn.BatchNorm1d(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        feat = self.features(x)
        feat = feat.view(B, -1)
        out = self.fc(feat)
        return self.norm(out)


# --------------------------------------------------------------------------- #
# Hybrid classifier – public API
# --------------------------------------------------------------------------- #

class HybridTransformerClassifier(nn.Module):
    """
    A transformer‑based classifier that can operate on text or image data.

    Parameters
    ----------
    front_end : Literal["text", "image"]
        Which front‑end to use. ``"text"`` uses an embedding layer and positional
        encoding; ``"image"`` uses a convolutional feature extractor.
    vocab_size : int, optional
        Vocabulary size for text input. Ignored when ``front_end="image"``.
    embed_dim : int
        Dimensionality of the token / feature representation.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Hidden dimension of the feed‑forward network.
    num_classes : int
        Number of target classes.
    dropout : float
        Drop‑out probability.
    use_quantum_attention : bool, default=False
        Flag retained for API compatibility. In the classical implementation
        this flag is ignored; the attention block is always classical.
    use_quantum_ffn : bool, default=False
        Flag retained for API compatibility. In the classical implementation
        this flag is ignored; the feed‑forward block is always classical.
    """
    def __init__(
        self,
        front_end: Literal["text", "image"],
        vocab_size: Optional[int] = None,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_blocks: int = 4,
        ffn_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        use_quantum_attention: bool = False,
        use_quantum_ffn: bool = False,
    ):
        super().__init__()
        self.front_end_type = front_end
        self.use_quantum_attention = use_quantum_attention
        self.use_quantum_ffn = use_quantum_ffn

        if front_end == "text":
            if vocab_size is None:
                raise ValueError("vocab_size must be provided for text front‑end")
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
            self.pos_encoder = PositionalEncoder(embed_dim)
        elif front_end == "image":
            self.front_end = ConvolutionalFeatureExtractor(output_dim=embed_dim)
        else:
            raise ValueError("front_end must be 'text' or 'image'")

        self.transformer_blocks = nn.ModuleList([
            TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_blocks)
        ])

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.front_end_type == "text":
            x = self.token_embedding(x)          # shape: (B, T, E)
            x = self.pos_encoder(x)
        else:  # image
            x = self.front_end(x)                # shape: (B, E)

        for block in self.transformer_blocks:
            x = block(x)

        # For text: mean over sequence; for image: already a single vector
        if self.front_end_type == "text":
            x = x.mean(dim=1)

        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "ConvolutionalFeatureExtractor",
    "HybridTransformerClassifier",
]
