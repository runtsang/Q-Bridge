"""Unified classical classifier with quantum‑inspired structure.

This module provides a `build_classifier_circuit` function that returns:
- a neural network implementing a depth‑controlled feed‑forward backbone,
- optional transformer encoder layers,
- a self‑attention style routing module,
- and metadata for weight bookkeeping and observables.

The API is kept identical to the original `QuantumClassifierModel.py` so that
existing pipelines can swap in the new implementation without changes.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Self‑attention routing – optional
# --------------------------------------------------------------------------- #
class _SelfAttentionRouter(nn.Module):
    """Compute a weighted sum of the input using a learned query/key/value projection."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn   = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)


# --------------------------------------------------------------------------- #
#  Transformer encoder block
# --------------------------------------------------------------------------- #
class _TransformerEncoderBlock(nn.Module):
    """Single transformer encoder with residual connections."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


# --------------------------------------------------------------------------- #
#  Unified classifier
# --------------------------------------------------------------------------- #
class UnifiedQuantumClassifier(nn.Module):
    """Depth‑controlled feed‑forward + optional transformer + optional self‑attention."""
    def __init__(
        self,
        num_features: int,
        depth: int,
        transformer_heads: int = 4,
        transformer_ffn: int = 16,
        transformer_layers: int = 1,
        dropout: float = 0.0,
        weight_trace: bool = False,
    ):
        super().__init__()
        self.weight_trace = weight_trace
        self.weight_sizes: List[int] = []

        # Optional self‑attention routing
        self.router = _SelfAttentionRouter(num_features) if weight_trace else None

        # Feed‑forward backbone
        layers = []
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            self.weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features

        # Optional transformer stack
        for _ in range(transformer_layers):
            block = _TransformerEncoderBlock(num_features, transformer_heads, transformer_ffn, dropout)
            layers.append(block)

        self.body = nn.Sequential(*layers)

        # Classifier head
        self.head = nn.Linear(num_features, 2)
        self.weight_sizes.append(self.head.weight.numel() + self.head.bias.numel())

        # Observables – trivial placeholder for compatibility
        self.observables = [0, 1] if weight_trace else [0, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.router:
            x = self.router(x)
        x = self.body(x)
        return self.head(x)


def build_classifier_circuit(
    num_features: int,
    depth: int,
    transformer_heads: int = 4,
    transformer_ffn: int = 16,
    transformer_layers: int = 1,
    dropout: float = 0.0,
    weight_trace: bool = False,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Build a classical classifier that mirrors the quantum helper interface.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector (same as qubit count in the quantum model).
    depth : int
        Number of linear‑ReLU blocks in the feed‑forward backbone.
    transformer_heads : int, optional
        Number of heads in the optional transformer encoder.
    transformer_ffn : int, optional
        Size of the internal feed‑forward layer inside each transformer block.
    transformer_layers : int, optional
        Number of transformer blocks stacked after the feed‑forward backbone.
    dropout : float, optional
        Dropout probability used inside the transformer blocks.
    weight_trace : bool, optional
        If True, a self‑attention router is inserted and weight bookkeeping
        is enabled for ablation studies.

    Returns
    -------
    nn.Module
        The constructed classifier.
    Iterable[int]
        Encoding mapping – list of input indices (identity for this implementation).
    Iterable[int]
        List of weight counts for each trainable linear layer.
    List[int]
        Placeholder observables list (kept for API compatibility).
    """
    model = UnifiedQuantumClassifier(
        num_features,
        depth,
        transformer_heads,
        transformer_ffn,
        transformer_layers,
        dropout,
        weight_trace,
    )
    encoding = list(range(num_features))
    weight_sizes = model.weight_sizes
    observables = model.observables
    return model, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit", "UnifiedQuantumClassifier"]
