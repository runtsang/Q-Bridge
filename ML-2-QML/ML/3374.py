"""Hybrid graph neural network module with classical operations and
self‑attention.

The module extends the original GraphQNN utilities by adding a
self‑attention layer that can be applied in a classical mode.  The
class `GraphQNNHybrid` exposes a unified API while delegating the
heavy lifting to the underlying classical implementations.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Generate a random weight matrix for a dense layer."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Create synthetic training pairs (x, Wx) for a given weight matrix."""
    data: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        x = torch.randn(weight.size(1), dtype=torch.float32)
        y = weight @ x
        data.append((x, y))
    return data


def random_network(qnn_arch: Sequence[int], samples: int):
    """Instantiate a random classical network and its training set."""
    weights: List[Tensor] = [
        _random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])
    ]
    target_weight = weights[-1]
    training_set = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_set, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Compute layer‑wise activations for a batch of samples."""
    activations: List[List[Tensor]] = []
    for x, _ in samples:
        layer_out = [x]
        cur = x
        for w in weights:
            cur = torch.tanh(w @ cur)
            layer_out.append(cur)
        activations.append(layer_out)
    return activations


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two normalized vectors."""
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
    """Build a weighted graph from state fidelities."""
    g = nx.Graph()
    g.add_nodes_from(range(len(states)))
    for (i, ai), (j, aj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(ai, aj)
        if fid >= threshold:
            g.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            g.add_edge(i, j, weight=secondary_weight)
    return g


class ClassicalSelfAttention:
    """Fast, NumPy‑based self‑attention block."""

    def __init__(self, embed_dim: int = 4):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        q = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        k = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        v = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(q @ k.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ v).numpy()


class GraphQNNHybrid:
    """Unified interface for classical GNN operations and optional self‑attention."""

    def __init__(self, qnn_arch: Sequence[int]):
        self.arch = list(qnn_arch)
        self.weights = [
            _random_linear(in_f, out_f)
            for in_f, out_f in zip(self.arch[:-1], self.arch[1:])
        ]
        self.attention = ClassicalSelfAttention(embed_dim=self.arch[-1])

    def feedforward(
        self, samples: Iterable[Tuple[Tensor, Tensor]]
    ) -> List[List[Tensor]]:
        return feedforward(self.arch, self.weights, samples)

    def fidelity_graph(
        self,
        states: Sequence[Tensor],
        *args,
        **kwargs,
    ) -> nx.Graph:
        return fidelity_adjacency(states, *args, **kwargs)

    def run_attention(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        return self.attention.run(rotation_params, entangle_params, inputs)


__all__ = [
    "GraphQNNHybrid",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
