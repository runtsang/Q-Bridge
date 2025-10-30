"""Combined classical transformer and fast estimator utilities.

This module extends the original QTransformerTorch implementation by adding
a FastEstimator wrapper that supports batched evaluation of scalar observables
and optional shot noise.  The core transformer blocks remain unchanged but
a new ``HybridTransformerEstimator`` class provides a unified API for
classical inference and noisy evaluation.
"""

from __future__ import annotations

import math
from typing import Optional, Iterable, List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
#  Classical transformer components (copied from QTransformerTorch.py)
# --------------------------------------------------------------------------- #

class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention implemented with PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out

class FeedForwardClassical(nn.Module):
    """Two‑layer feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockClassical(nn.Module):
    """A single transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn   = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
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
        return x + self.pe[:, :x.size(1)]

class TextClassifier(nn.Module):
    """Transformer‑based text classifier."""
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
        self.pos_embedding   = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

    # --------------------------------------------------------------------- #
    #  New API: evaluate observables directly on the logits
    # --------------------------------------------------------------------- #
    def evaluate_observables(self,
                             x: torch.Tensor,
                             observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]]
                             ) -> List[float]:
        """Return a list of scalar values for each observable."""
        logits = self.forward(x)
        results: List[float] = []
        for obs in observables:
            val = obs(logits)
            if isinstance(val, torch.Tensor):
                results.append(float(val.mean().item()))
            else:
                results.append(float(val))
        return results

# --------------------------------------------------------------------------- #
#  FastEstimator wrapper for batched evaluation with optional shot noise
# --------------------------------------------------------------------------- #

class FastEstimator:
    """Evaluate a model for a list of parameter sets and scalar observables.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.  It must accept a ``torch.LongTensor`` of token
        indices and return a tensor of logits.
    shots : int | None, optional
        If provided, Gaussian noise with variance ``1/shots`` is added to each
        observable to mimic shot noise.
    seed : int | None, optional
        Random seed for reproducible noise.
    """
    def __init__(self,
                 model: nn.Module,
                 shots: Optional[int] = None,
                 seed: Optional[int] = None) -> None:
        self.model = model
        self.shots = shots
        self.seed = seed
        self._rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: List[List[int]]) -> List[List[float]]:
        """Return a matrix of observable values.

        Each row corresponds to a parameter set; each column to an observable.
        """
        observables = list(observables) or [lambda logits: logits.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                # Convert to tensor of token indices
                inputs = torch.as_tensor(params, dtype=torch.long)
                if inputs.ndim == 1:
                    inputs = inputs.unsqueeze(0)
                logits = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(logits)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().item())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                # Add shot‑noise if requested
                if self.shots is not None:
                    row = [float(self._rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
                results.append(row)
        return results

# --------------------------------------------------------------------------- #
#  Unified estimator class that bundles the transformer and the FastEstimator
# --------------------------------------------------------------------------- #

class HybridTransformerEstimator(FastEstimator):
    """Convenience class that builds a transformer and exposes a FastEstimator API."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 shots: Optional[int] = None,
                 seed: Optional[int] = None) -> None:
        model = TextClassifier(vocab_size, embed_dim, num_heads, num_blocks,
                               ffn_dim, num_classes, dropout)
        super().__init__(model, shots, seed)

__all__ = [
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "TextClassifier",
    "FastEstimator",
    "HybridTransformerEstimator",
]
