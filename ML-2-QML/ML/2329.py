"""Classical transformer implementation with optional SelfAttention‑based attention.

This module mirrors the original QTransformerTorch API but adds a lightweight
SelfAttention block inspired by the SelfAttention.py seed.  The
`attention_type` argument selects between the standard PyTorch
nn.MultiheadAttention and the SelfAttention variant.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Self‑attention helper
# --------------------------------------------------------------------------- #
class SelfAttention:
    """Naïve self‑attention using NumPy, mirroring the SelfAttention seed.

    The implementation is intentionally simple: it projects the input
    through rotation and entangle matrices, computes a soft‑max over the
    resulting scores, and returns the weighted sum of the original tokens.
    """
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def __call__(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        # Linear projections
        query = inputs @ rotation_params.reshape(self.embed_dim, -1)
        key   = inputs @ entangle_params.reshape(self.embed_dim, -1)
        # Attention scores
        scores = np.exp(query @ key.T / math.sqrt(self.embed_dim))
        scores /= scores.sum(axis=-1, keepdims=True)
        # Weighted sum
        return scores @ inputs

# --------------------------------------------------------------------------- #
# Attention modules
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Base class for all attention variants."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        raise NotImplementedError

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention using torch.nn.MultiheadAttention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output

class MultiHeadAttentionSelf(MultiHeadAttentionBase):
    """Multi‑head attention that delegates the score computation to the
    SelfAttention helper.  Each head uses the same rotation/entangle
    parameters for simplicity."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.head_dim = embed_dim // num_heads
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.self_attn = SelfAttention(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        batch, seq, _ = x.shape
        proj = self.proj(x)                     # (B, S, E)
        proj = proj.view(batch, seq, self.num_heads, self.head_dim)
        proj = proj.transpose(1, 2)              # (B, H, S, D)
        outputs = []
        for h in range(self.num_heads):
            head = proj[:, h]                   # (B, S, D)
            # Flatten batch and seq for the NumPy helper
            flat = head.reshape(-1, self.head_dim).cpu().numpy()
            # Random parameters – in practice these would be learned
            rot = np.random.randn(self.head_dim * 3)
            ent = np.random.randn(self.head_dim - 1)
            out = self.self_attn(rot, ent, flat)
            out = torch.from_numpy(out).float().reshape(batch, seq, self.head_dim)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)       # (B, H, S, D)
        out = out.transpose(1, 2).contiguous()  # (B, S, H, D)
        out = out.view(batch, seq, self.embed_dim)
        return self.dropout(out)

# --------------------------------------------------------------------------- #
# Feed‑forward modules
# --------------------------------------------------------------------------- #
class FeedForwardBase(nn.Module):
    """Base class for feed‑forward blocks."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    """Standard two‑layer MLP."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# --------------------------------------------------------------------------- #
# Transformer block
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    """Base class for a transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    """Standard transformer block."""
    def __init__(self, embed_dim: int, num_heads: int,
                 ffn_dim: int, dropout: float = 0.1,
                 attention_type: str = "nn"):
        super().__init__(embed_dim, num_heads, dropout)
        if attention_type == "self":
            self.attn = MultiHeadAttentionSelf(embed_dim, num_heads, dropout)
        else:
            self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor):
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) *
            (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        return x + self.pe[:, :x.size(1)]

# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #
class QTransformerTorch(nn.Module):
    """Hybrid transformer that can switch between classical and
    SelfAttention‑based attention.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    embed_dim : int
        Dimensionality of token embeddings.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer layers.
    ffn_dim : int
        Hidden size of the feed‑forward network.
    num_classes : int
        Number of target classes.
    dropout : float, optional
        Drop‑out probability.
    attention_type : str, optional
        ``"nn"`` for the standard PyTorch implementation,
        ``"self"`` to use the SelfAttention helper.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        attention_type: str = "nn",
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlockClassical(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    dropout,
                    attention_type=attention_type,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim,
            num_classes if num_classes > 2 else 1,
        )

    def forward(self, x: torch.Tensor):
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for block in self.layers:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "SelfAttention",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionSelf",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "QTransformerTorch",
]
