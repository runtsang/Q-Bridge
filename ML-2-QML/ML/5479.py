"""Hybrid graph‑transformer architecture combining classical transformer, graph fidelity, and fast estimation."""

from __future__ import annotations

import math
import itertools
from typing import Sequence, Iterable, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class MultiHeadAttention(nn.Module):
    """Standard multi‑head attention with optional graph mask."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # mask shape (batch, seq_len) where 0 means masked
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out


class FeedForward(nn.Module):
    """Two‑layer MLP with dropout."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Canonical transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return cosine similarity matrix between two sets of vectors."""
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    return torch.mm(a_norm, b_norm.t())


def fidelity_adjacency(embeddings: torch.Tensor, threshold: float,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build graph where nodes are embeddings and edges weighted by cosine similarity."""
    n = embeddings.size(0)
    sim = cosine_similarity(embeddings, embeddings)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            fid = sim[i, j].item()
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph


class FastBaseEstimator:
    """Batch evaluation of a model with optional shot noise."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(self,
                 observables: Iterable[ScalarObservable],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inp = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
                out = self.model(inp)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot noise to deterministic outputs."""
    def evaluate(self,
                 observables: Iterable[ScalarObservable],
                 parameter_sets: Sequence[Sequence[float]],
                 *,
                 shots: int | None = None,
                 seed: int | None = None) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


class HybridGraphTransformer(nn.Module):
    """Classical transformer operating on graph‑structured data with fidelity‑based adjacency."""
    def __init__(self,
                 num_nodes: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 threshold: float = 0.8,
                 secondary: float | None = None,
                 secondary_weight: float = 0.5,
                 dropout: float = 0.1):
        super().__init__()
        self.node_emb = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        self.threshold = threshold
        self.secondary = secondary
        self.secondary_weight = secondary_weight

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            batch: shape (batch, num_nodes) containing indices of nodes
        """
        # gather node embeddings
        x = self.node_emb[batch]  # (batch, num_nodes, embed_dim)
        # build adjacency for each example
        adjacency = torch.stack([fidelity_adjacency(x[i], self.threshold,
                                                    self.secondary,
                                                    self.secondary_weight)
                                for i in range(batch.size(0))], dim=0)
        # create mask: 1 indicates valid, 0 masked
        mask = torch.stack([torch.tensor([list(adj.adj[v]) for v in range(x.size(1))], dtype=torch.bool)
                            for adj in adjacency], dim=0)
        # forward through transformer
        x = self.transformer(x, mask=mask)
        # pool and classify
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


__all__ = [
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlock",
    "fidelity_adjacency",
    "FastBaseEstimator",
    "FastEstimator",
    "HybridGraphTransformer",
]
