"""Pure‑Python transformer with optional quantum hooks for experimentation.

The module contains a classical implementation of the transformer blocks
and a lightweight `SelfAttentionHelper` that mimics the quantum
self‑attention interface from the original repository.  The
`TextClassifier` class accepts optional `use_quantum` flag; when
`True` the class will attempt to use a quantum backend if available,
otherwise it falls back to the classical path.  This design preserves
API compatibility with the original `QTransformerTorch.py` while
providing an explicit entry point for hybrid experiments.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------

class SelfAttentionHelper:
    """Classical mimic of the quantum‑style self‑attention circuit.

    The helper accepts rotation and entanglement parameters and
    produces an attention matrix using the same mathematical form
    as the Qiskit implementation.  It is intentionally lightweight
    and fully compatible with NumPy / PyTorch tensors.
    """
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key   = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / math.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

# ---------------------------------------------------------------------------

class MultiHeadAttentionBase(nn.Module):
    """Shared interface for attention implementations."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout   = nn.Dropout(dropout)
        self.d_k = embed_dim // num_heads

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

    def downstream(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.separate_heads(query)
        k = self.separate_heads(key)
        v = self.separate_heads(value)
        out, _ = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(query.size(0), -1, self.embed_dim)

# ---------------------------------------------------------------------------

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        return self.downstream(q, k, v, mask)

# ---------------------------------------------------------------------------

class MultiHeadAttentionHybrid(MultiHeadAttentionBase):
    """Hybrid attention that delegates to a quantum backend if available.

    In the pure‑classical build this class simply forwards to
    `MultiHeadAttentionClassical`.  The quantum implementation is
    provided in the QML module and is lazily imported to keep the
    classical build lightweight.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_quantum: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.use_quantum = use_quantum
        if not use_quantum:
            self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        else:
            # pragma: no cover
            # In the hybrid build, the quantum implementation will be
            # provided by the QML module.  Importing lazily prevents a
            # hard dependency on TorchQuantum.
            from.qml import MultiHeadAttentionQuantum  # type: ignore
            self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.attn(x, mask)

# ---------------------------------------------------------------------------

class FeedForwardBase(nn.Module):
    """Base class for feed‑forward blocks."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim   = ffn_dim
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

# ---------------------------------------------------------------------------

class FeedForwardClassical(FeedForwardBase):
    """Standard two‑layer feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# ---------------------------------------------------------------------------

class FeedForwardHybrid(FeedForwardBase):
    """Hybrid feed‑forward that optionally uses a quantum module."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1, use_quantum: bool = False) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.use_quantum = use_quantum
        if not use_quantum:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        else:
            # pragma: no cover
            from.qml import FeedForwardQuantum  # type: ignore
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

# ---------------------------------------------------------------------------

class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1   = nn.LayerNorm(embed_dim)
        self.norm2   = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

# ---------------------------------------------------------------------------

class TransformerBlockClassical(TransformerBlockBase):
    """Pure‑classical transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# ---------------------------------------------------------------------------

class TransformerBlockHybrid(TransformerBlockBase):
    """Hybrid transformer block that can use quantum sub‑modules."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, use_quantum: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionHybrid(embed_dim, num_heads, dropout, use_quantum)
        self.ffn = FeedForwardHybrid(embed_dim, ffn_dim, dropout, use_quantum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------

class TextClassifier(nn.Module):
    """Hybrid text classifier that supports optional quantum sub‑modules.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    embed_dim : int
        Dimensionality of token embeddings.
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
    use_quantum : bool, optional
        If True, the classifier will attempt to use the quantum
        implementations of attention and feed‑forward.  The quantum
        implementation is provided in the QML module and is lazily
        imported to keep the classical build lightweight.
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
        use_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding   = PositionalEncoder(embed_dim)
        self.use_quantum     = use_quantum
        block_cls = TransformerBlockHybrid if use_quantum else TransformerBlockClassical
        self.transformers = nn.Sequential(
            *[block_cls(embed_dim, num_heads, ffn_dim, dropout, use_quantum) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

__all__ = [
    "SelfAttentionHelper",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionHybrid",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardHybrid",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockHybrid",
    "PositionalEncoder",
    "TextClassifier",
]
