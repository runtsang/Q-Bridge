"""SharedClassName – classical implementation.

This module provides a single entry point that exposes all the classical
capabilities seen in the original seeds: random graph‑based networks,
forward propagation, fidelity‑based adjacency graphs, a lightweight
estimator, transformer blocks, and a quanvolution classifier.  All
utilities are fully self‑contained and importable without external
dependencies beyond PyTorch and NetworkX.
"""

from __future__ import annotations

import itertools
import math
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
#  Graph‑based utilities (from GraphQNN.py)
# --------------------------------------------------------------------------- #
def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Generate a random linear network and training data."""
    weights = [torch.randn(out, in_, dtype=torch.float32) for in_, out in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]
    training_data = [(torch.randn(target_weight.size(1)), target_weight @ torch.randn(target_weight.size(1))) for _ in range(samples)]
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
    """Forward pass through a purely linear network with tanh activations."""
    activations: List[List[Tensor]] = []
    for features, _ in samples:
        current = features
        layer_acts = [features]
        for w in weights:
            current = torch.tanh(w @ current)
            layer_acts.append(current)
        activations.append(layer_acts)
    return activations


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two classical feature vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted graph where edges represent state similarity."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#  Estimator utilities (from FastBaseEstimator.py)
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate a PyTorch model for a batch of parameter sets."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(self, observables: Iterable[Callable[[Tensor], Tensor | float]], parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32)
                if inputs.ndim == 1:
                    inputs = inputs.unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        row.append(float(val.mean().cpu()))
                    else:
                        row.append(float(val))
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot noise to deterministic results."""

    def evaluate(self, observables: Iterable[Callable[[Tensor], Tensor | float]], parameter_sets: Sequence[Sequence[float]], *, shots: int | None = None, seed: int | None = None) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = [[float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row] for row in raw]
        return noisy


# --------------------------------------------------------------------------- #
#  Transformer utilities (from QTransformerTorch.py)
# --------------------------------------------------------------------------- #
class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out


class FeedForwardClassical(nn.Module):
    """Two‑layer MLP with dropout."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockClassical(nn.Module):
    """Single transformer block."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1)]


class TextClassifier(nn.Module):
    """Transformer‑based text classifier."""

    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_blocks: int,
                 ffn_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: Tensor) -> Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


# --------------------------------------------------------------------------- #
#  Quanvolution utilities (from Quanvolution.py)
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    """Classical 2×2 filter that mimics a quantum kernel."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """Simple classifier that uses the quanvolution filter."""

    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


# --------------------------------------------------------------------------- #
#  SharedClassName – public API
# --------------------------------------------------------------------------- #
class SharedClassName:
    """Public façade that bundles all the utilities above."""

    # expose utilities as class attributes for convenience
    random_network = staticmethod(random_network)
    feedforward = staticmethod(feedforward)
    state_fidelity = staticmethod(state_fidelity)
    fidelity_adjacency = staticmethod(fidelity_adjacency)

    # estimator classes
    FastBaseEstimator = FastBaseEstimator
    FastEstimator = FastEstimator

    # transformer components
    MultiHeadAttentionClassical = MultiHeadAttentionClassical
    FeedForwardClassical = FeedForwardClassical
    TransformerBlockClassical = TransformerBlockClassical
    PositionalEncoder = PositionalEncoder
    TextClassifier = TextClassifier

    # quanvolution components
    QuanvolutionFilter = QuanvolutionFilter
    QuanvolutionClassifier = QuanvolutionClassifier


__all__ = [
    "SharedClassName",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "FastBaseEstimator",
    "FastEstimator",
    "MultiHeadAttentionClassical",
    "FeedForwardClassical",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "TextClassifier",
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
]
