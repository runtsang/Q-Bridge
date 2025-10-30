"""Hybrid transformer classifier with optional classical self‑attention.

This module implements a fully classical transformer that can be
configured to use either a dense self‑attention mechanism or a
parameterised matrix‑based self‑attention layer.  It also ships
with a lightweight regression dataset that mirrors the quantum
example, facilitating side‑by‑side performance studies.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Data utilities
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) where X ∈ ℝ^{samples×num_features} and
    y = sin(sum(X)) + 0.1*cos(2*sum(X)).
    """
    X = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = X.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return X, y.astype(np.float32)


class SuperpositionDataset(Dataset):
    """Simple regression dataset based on the superposition generator."""
    def __init__(self, samples: int, num_features: int):
        self.X, self.y = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.X)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {"states": torch.tensor(self.X[idx], dtype=torch.float32),
                "target": torch.tensor(self.y[idx], dtype=torch.float32)}


# --------------------------------------------------------------------------- #
# Self‑attention utilities
# --------------------------------------------------------------------------- #

class ClassicalSelfAttention:
    """Dense matrix self‑attention with rotation & entanglement params."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        rot = rotation_params.reshape(self.embed_dim, -1)
        ent = entangle_params.reshape(self.embed_dim, -1)
        query = torch.as_tensor(inputs @ rot, dtype=torch.float32)
        key   = torch.as_tensor(inputs @ ent, dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / math.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


# --------------------------------------------------------------------------- #
# Classical transformer blocks
# --------------------------------------------------------------------------- #

class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionDense(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with nn.MultiheadAttention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return out


class MultiHeadAttentionSelf(MultiHeadAttentionBase):
    """Self‑attention using the ClassicalSelfAttention matrix."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = ClassicalSelfAttention(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # For simplicity we ignore the mask; real use‑case would split batch.
        return torch.from_numpy(
            self.attn.run(np.random.randn(self.embed_dim, self.num_heads),
                          np.random.randn(self.embed_dim, self.num_heads),
                          x.detach().cpu().numpy())
        ).to(x.device)


class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardDense(FeedForwardBase):
    """Two‑layer MLP with ReLU."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockDense(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int,
                 ffn_dim: int, dropout: float = 0.1,
                 use_self_attention: bool = False):
        super().__init__(embed_dim, num_heads, dropout)
        attn_cls = MultiHeadAttentionSelf if use_self_attention else MultiHeadAttentionDense
        self.attn = attn_cls(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardDense(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class HybridTransformerClassifier(nn.Module):
    """Fully classical transformer classifier with optional self‑attention."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_self_attention: bool = False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(*[
            TransformerBlockDense(embed_dim, num_heads, ffn_dim,
                                 dropout, use_self_attention)
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


__all__ = [
    "HybridTransformerClassifier",
    "SuperpositionDataset",
    "generate_superposition_data",
]
