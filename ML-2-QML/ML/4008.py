"""HybridTransformer – classical implementation with API parity to the original QTransformerTorch.

This module implements a fully‑classical transformer encoder that can serve as a drop‑in
replacement for the original TextClassifier.  It supports an optional flag *use_quantum*
for API compatibility – the flag is ignored in the classical implementation but
allows the quantum module to raise a clear error if the flag is mis‑used.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Core components – classical
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

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


class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention using PyTorch's implementation."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # type: ignore[override]
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class FeedForwardClassical(nn.Module):
    """Two‑layer MLP used in transformer blocks."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockClassical(nn.Module):
    """Transformer encoder block built from classical sub‑modules."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
#  HybridTransformer – public API
# --------------------------------------------------------------------------- #
class HybridTransformer(nn.Module):
    """Hybrid transformer that defaults to a fully‑classical implementation.

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
        Hidden size of the feed‑forward network.
    num_classes : int
        Number of output classes (>=2 for classification, <2 for regression).
    dropout : float, default 0.1
        Dropout probability.
    use_quantum : bool, default False
        Flag for API parity – ignored in the classical implementation.
    q_config : dict | None, default None
        Quantum configuration placeholder for the QML module.
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
        q_config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.use_quantum = use_quantum  # stored for API parity
        if self.use_quantum:
            # Warn the user – classical code cannot execute quantum operations
            import warnings

            warnings.warn(
                "HybridTransformer instantiated with use_quantum=True "
                "in the classical module; falling back to classical paths."
            )

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        blocks = [
            TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_blocks)
        ]
        self.transformer = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass for classification or regression."""
        tokens = self.token_embedding(x)
        x = self.positional_encoding(tokens)
        x = self.transformer(x)
        x = x.mean(dim=1)  # CLS‑like global pooling
        x = self.dropout(x)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
#  Utility – regression helper
# --------------------------------------------------------------------------- #
def EstimatorQNN() -> nn.Module:
    """Return a tiny fully‑connected regression network (mirror of the QML example)."""

    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(inputs)

    return EstimatorNN()


# --------------------------------------------------------------------------- #
#  Backwards compatibility alias
# --------------------------------------------------------------------------- #
TextClassifier = HybridTransformer

__all__ = [
    "PositionalEncoding",
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "TransformerBlockClassical",
    "HybridTransformer",
    "TextClassifier",
    "EstimatorQNN",
]
