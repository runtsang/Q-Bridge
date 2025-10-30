"""Hybrid graph‑neural‑network module with transformer propagation.

The API mirrors the original GraphQNN but now supports a
classical transformer‑based propagation across the graph.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from.Conv import Conv  # local convolution filter

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def state_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class TransformerBlock(nn.Module):
    """A single transformer layer with multi‑head attention and FFN."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch, seq_len, embed_dim)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class GraphQNNML(nn.Module):
    """Classical graph‑neural‑network with transformer propagation."""

    def __init__(
        self,
        arch: Sequence[int],
        dropout: float = 0.1,
        num_heads: int = 4,
        ffn_dim: int = 64,
        adjacency_threshold: float = 0.8,
        secondary_threshold: float | None = None,
    ):
        super().__init__()
        self.arch = list(arch)
        self.dropout = dropout
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.adjacency_threshold = adjacency_threshold
        self.secondary_threshold = secondary_threshold

        # linear layers for each graph layer
        self.linears = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(self.arch[:-1], self.arch[1:])]
        )

        # transformer blocks applied after each linear layer
        self.transformers = nn.ModuleList(
            [
                TransformerBlock(self.arch[i + 1], self.num_heads, self.ffn_dim, self.dropout)
                for i in range(len(self.arch) - 1)
            ]
        )

        # local convolution filter
        self.conv = Conv()

    def build_graph(self, states: Sequence[Tensor]) -> nx.Graph:
        """Create a graph from state‑fidelity of node states."""
        return fidelity_adjacency(
            states,
            self.adjacency_threshold,
            secondary=self.secondary_threshold,
        )

    def feedforward(
        self,
        node_features: Tensor,
    ) -> List[Tensor]:
        """
        node_features: shape (num_nodes, feature_dim)
        Returns a list of activations per graph layer.
        """
        activations: List[Tensor] = [node_features]
        x = node_features.unsqueeze(0)  # batch dimension for attention
        for linear, transformer in zip(self.linears, self.transformers):
            # local convolution on each node
            conv_out = torch.stack(
                [torch.tensor(self.conv.run(n.unsqueeze(0).numpy())) for n in x.squeeze(0)]
            ).unsqueeze(1)  # shape (1, num_nodes, 1)
            x = torch.cat([x, conv_out], dim=-1)  # augment features
            x = linear(x)  # shape (1, num_nodes, out_dim)
            x = transformer(x)  # transformer across nodes
            activations.append(x.squeeze(0))
        return activations

    def random_network(self, samples: int = 10):
        """Convenience wrapper around the global random_network helper."""
        return random_network(self.arch, samples)

    def state_fidelity(self, a: Tensor, b: Tensor) -> float:
        return state_fidelity(a, b)

    def fidelity_adjacency(self, states: Sequence[Tensor], threshold: float) -> nx.Graph:
        return fidelity_adjacency(states, threshold)

__all__ = [
    "GraphQNNML",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
