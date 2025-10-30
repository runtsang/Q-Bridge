"""HybridTransformer: Classical transformer with optional quantum-aware API.

This module implements a self‑contained, fully classical transformer
architecture that mirrors the quantum‑oriented API of the original
seed.  It is intentionally lightweight yet fully compatible with the
anchor reference path ``QTransformerTorch.py`` – the class names,
constructor signatures and attribute names are preserved so that
downstream code can import without modification.

Key design choices

* ``HybridTransformer`` exposes a ``use_quantum`` flag.  When set to
  ``True`` the constructor raises an informative error because the
  quantum implementation lives in the companion QML module.  This
  explicit guard keeps the classical module free of any quantum
  dependencies while still presenting a unified public API.

* The module re‑exports the building blocks from the seed
  implementation (``MultiHeadAttentionClassical``,
  ``FeedForwardClassical``, ``TransformerBlockClassical``) so that
  custom experiments can reuse them.

* A small ``FastBaseEstimator`` helper is re‑implemented to evaluate
  the model on batches of parameters in a vectorised fashion.  It
  mimics the interface of the QML version but operates on tensors,
  making it convenient for quick sanity checks.

The module is fully self‑contained and can be dropped into any
PyTorch‑based training loop.
"""

from __future__ import annotations

import math
from typing import Optional, Iterable, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Core building blocks
# --------------------------------------------------------------------------- #

class MultiHeadAttentionBase(nn.Module):
    """
    Shared logic for attention layers.  The base class is kept for API
    compatibility but is not intended to be instantiated directly.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.linear_q(x).view(batch, seq_len, self.num_heads, -1).transpose(1, 2)
        k = self.linear_k(x).view(batch, seq_len, self.num_heads, -1).transpose(1, 2)
        v = self.linear_v(x).view(batch, seq_len, self.num_heads, -1).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(out)


class FeedForwardBase(nn.Module):
    """Shared interface for feed‑forward layers."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)


class TransformerBlockClassical(TransformerBlockBase):
    """Purely classical transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
#  Positional encoding
# --------------------------------------------------------------------------- #

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding as used in the original transformer."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32)
                             * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
#  Fast estimator utilities (light‑weight re‑implementation)
# --------------------------------------------------------------------------- #

def _ensure_batch(values: Iterable[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Evaluate a PyTorch model for a batch of input parameters."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(self, observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: Iterable[Sequence[float]]) -> List[List[float]]:
        """Return a list of rows, each row containing the observable values for a parameter set."""
        obs = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in obs:
                    val = observable(outputs)
                    if isinstance(val, torch.Tensor):
                        row.append(float(val.mean().cpu()))
                    else:
                        row.append(float(val))
                results.append(row)
        return results


# --------------------------------------------------------------------------- #
#  Hybrid model
# --------------------------------------------------------------------------- #

class HybridTransformer(nn.Module):
    """
    A transformer that mirrors the quantum API but remains purely classical.
    The constructor accepts a ``use_quantum`` flag; when ``True`` an
    informative error is raised, directing the user to the quantum
    companion ``HybridTransformerQuantum`` defined in ``qml_code``.
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
        if use_quantum:
            raise RuntimeError(
                "HybridTransformer is the classical variant.  Use "
                "HybridTransformerQuantum from the QML module for quantum "
                "support."
            )
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "FastBaseEstimator",
    "HybridTransformer",
]
