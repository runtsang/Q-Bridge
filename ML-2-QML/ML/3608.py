"""UnifiedSelfAttentionTransformer – Classical implementation with optional quantum plug‑in.

The class implements a transformer‑style text classifier that can be configured to use quantum sub‑modules for the attention and feed‑forward layers.  The quantum modules are supplied via a dictionary and are completely isolated from the classical core, allowing the model to behave exactly like a standard transformer when no quantum config is given.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Classical self‑attention helper – used by the hybrid attention layer.
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention:
    """Pure‑tensor self‑attention that mirrors the original seed."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Compute a weighted sum of values (values = input)."""
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        # broadcasting: shape (B, E)
        if query.ndim == 1:
            query = query.unsqueeze(0)
        if key.ndim == 1:
            key = key.unsqueeze(0)
        scores = torch.softmax(query @ key.T / math.sqrt(self.embed_dim), dim=-1)
        return (scores @ torch.as_tensor(inputs, dtype=torch.float32)).numpy()

# --------------------------------------------------------------------------- #
# Classical multi‑head attention.
# --------------------------------------------------------------------------- #
class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention implemented classically."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_linear = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        # projections
        k = self.k_linear(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        q = self.q_linear(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        scores = self.dropout(F.softmax(scores, dim=-1))
        attn = torch.matmul(scores, v)
        # concatenate heads
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_linear(attn)

# --------------------------------------------------------------------------- #
# Classical feed‑forward network.
# --------------------------------------------------------------------------- #
class FeedForwardClassical(nn.Module):
    """Two‑layer perceptron feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# --------------------------------------------------------------------------- #
# Positional encoding.
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
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
# Transformer block – can optionally delegate to a quantum module.
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    """Residual transformer block with optional quantum attention/FF."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        quantum_config: dict | None = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # quantum_config may contain 'attn' and 'ffn' keys providing callable modules
        if quantum_config and "attn" in quantum_config:
            self.attn = quantum_config["attn"]
        else:
            self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)

        if quantum_config and "ffn" in quantum_config:
            self.ffn = quantum_config["ffn"]
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# Text classifier that stitches everything together.
# --------------------------------------------------------------------------- #
class UnifiedSelfAttentionTransformer(nn.Module):
    """Hybrid transformer‑style classifier.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embed_dim : int
        Embedding dimension.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Hidden dimension of the feed‑forward network.
    num_classes : int
        Number of output classes.
    dropout : float, optional
        Dropout probability.
    quantum_config : dict, optional
        Dictionary mapping component names ('attn', 'ffn') to quantum
        modules that implement the same interface as their classical
        counterparts.
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
        quantum_config: dict | None = None,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    dropout,
                    quantum_config=quantum_config,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

__all__ = ["UnifiedSelfAttentionTransformer", "ClassicalSelfAttention"]
