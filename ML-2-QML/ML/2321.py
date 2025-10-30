"""Hybrid self‑attention module with classical attention and graph utilities.

The class HybridSelfAttentionQNN implements:
- Multi‑head self‑attention with learnable linear projections.
- Fidelity‑based graph construction from the attention‑weighted states.
- Optional feedforward through a classical network of linear layers.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch

Tensor = torch.Tensor


# ----------------------------------------------------------------------------- #
#  Classical utilities – adapted from GraphQNN.py
# ----------------------------------------------------------------------------- #

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
    for in_features, out_features in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_features, out_features))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


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


# ----------------------------------------------------------------------------- #
#  HybridSelfAttentionQNN – classical implementation
# ----------------------------------------------------------------------------- #

class HybridSelfAttentionQNN:
    """Hybrid classical self‑attention network with graph utilities.

    Parameters
    ----------
    embed_dim : int
        Dimension of the input embeddings.
    num_heads : int
        Number of attention heads.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Learnable linear projections
        self.W_q = torch.nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_k = torch.nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_v = torch.nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_o = torch.nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, x: Tensor) -> Tensor:
        """Compute multi‑head self‑attention and return context."""
        batch, seq_len, _ = x.shape
        Q = torch.matmul(x, self.W_q)
        K = torch.matmul(x, self.W_k)
        V = torch.matmul(x, self.W_v)

        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.view(batch, seq_len, self.embed_dim)
        return torch.matmul(context, self.W_o)

    def build_graph(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
    ) -> nx.Graph:
        """Construct a fidelity‑based adjacency graph from a set of state vectors."""
        return fidelity_adjacency(
            states, threshold, secondary=secondary, secondary_weight=0.5
        )

    def feedforward(
        self,
        qnn_arch: Sequence[int],
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Run a classical feed‑forward network (for comparison)."""
        return feedforward(qnn_arch, weights, samples)

    def random_network(self, qnn_arch: Sequence[int], samples: int):
        """Generate a random network and training data."""
        return random_network(qnn_arch, samples)

    def random_training_data(self, weight: Tensor, samples: int):
        """Generate random training data for a given weight matrix."""
        return random_training_data(weight, samples)

    def state_fidelity(self, a: Tensor, b: Tensor) -> float:
        """Return fidelity between two state vectors."""
        return state_fidelity(a, b)

    def __repr__(self) -> str:
        return (
            f"HybridSelfAttentionQNN(embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads})"
        )
