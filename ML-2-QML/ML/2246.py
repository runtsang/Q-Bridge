"""Unified QCNN‑Transformer (classical implementation).

The model consists of two stages:
1.  A QCNN‑style feature extractor built from fully‑connected layers that mimic the convolution‑pool pattern of the original QCNN paper.
2.  A transformer encoder that can be configured to use either classical or quantum sub‑modules.

The `UnifiedQCNNTransformer` class is fully compatible with the original `QCNN` API and exposes a `QCNN()` factory for quick instantiation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Classical QCNN‑style feature extractor
# --------------------------------------------------------------------------- #
class _QCNNFeatureExtractor(nn.Module):
    """Fully‑connected stack that mimics the QCNN convolution‑pool pattern.

    The layer sizes are taken from the original QCNN paper (8→16→12→8→4→2→1).
    Each “convolution” is a linear layer followed by a tanh, and each “pool”
    is a linear layer that reduces dimension.  The output is a 1‑dimensional
    representation that can be fed into a transformer encoder.
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.pool3 = nn.Sequential(nn.Linear(4, 2), nn.Tanh())
        self.pool4 = nn.Sequential(nn.Linear(2, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.pool4(x))

# --------------------------------------------------------------------------- #
# 2. Classical transformer components
# --------------------------------------------------------------------------- #
class _MultiHeadAttentionBase(nn.Module):
    """Shared logic for attention layers."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.size()
        return x.view(b, t, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, query: torch.Tensor, key: torch.Tensor,
                  value: torch.Tensor, mask: Optional[torch.Tensor] = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        return torch.matmul(scores, value), scores

class _MultiHeadAttentionClassical(_MultiHeadAttentionBase):
    """Standard multi‑head attention implemented in PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, _, embed_dim = x.size()
        if embed_dim!= self.embed_dim:
            raise ValueError("Input embedding mismatch")
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        q, k, v = self.separate_heads(q), self.separate_heads(k), self.separate_heads(v)
        out, _ = self.attention(q, k, v, mask)
        return self.combine_heads(out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim))

class _FeedForwardBase(nn.Module):
    """Base for feed‑forward networks."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

class _FeedForwardClassical(_FeedForwardBase):
    """Two‑layer perceptron feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class _TransformerBlockClassical(nn.Module):
    """Transformer block with classical attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = _MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = _FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
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

class TextClassifier(nn.Module):
    """Transformer‑based classifier that can be composed with QCNN features."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[ _TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
               for _ in range(num_blocks) ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

class UnifiedQCNNTransformer(nn.Module):
    """Combines a QCNN‑style feature extractor with a transformer encoder.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary (used only for the transformer part).
    embed_dim : int
        Dimensionality of the transformer embeddings.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Width of the feed‑forward layers.
    num_classes : int
        Number of output classes.
    dropout : float, optional
        Drop‑out probability.
    use_quantum : bool, optional
        If True, the transformer blocks will be replaced by their quantum
        counterparts (see the :mod:`qml` module).  The classical model
        remains unchanged.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_quantum: bool = False) -> None:
        super().__init__()
        self.feature_extractor = _QCNNFeatureExtractor()
        if use_quantum:
            raise NotImplementedError("Quantum variant requires the qml module")
        self.transformer = _TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is assumed to be a batch of token indices
        features = self.feature_extractor(x.float())
        # Expand features to match sequence length for transformer
        seq = features.unsqueeze(1)  # (batch, 1, 1)
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformer(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

def QCNN() -> UnifiedQCNNTransformer:
    """Convenience factory matching the original QCNN API."""
    return UnifiedQCNNTransformer(
        vocab_size=10000,
        embed_dim=16,
        num_heads=4,
        num_blocks=2,
        ffn_dim=32,
        num_classes=2,
        dropout=0.1,
        use_quantum=False
    )

__all__ = [
    "QCNN",
    "UnifiedQCNNTransformer",
    "_QCNNFeatureExtractor",
    "_TransformerBlockClassical",
    "TextClassifier",
]
