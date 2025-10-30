"""Hybrid graph neural network utilities with classical interface.

This module augments the original GraphQNN utilities by wrapping them
into a class `GraphQNN__gen318`.  The class exposes methods for
building a random network, running a forward pass, creating a
fidelity‑based adjacency graph, and generating random training data.
The API is intentionally minimal to keep the training pipeline
out of scope while providing a clear entry point for downstream
experiments.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate `samples` pairs (x, y) where y = weight @ x."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Return architecture, list of weight matrices, training data, and target weight."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Return activations per layer for each sample."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations: List[Tensor] = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two classical state vectors."""
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
    """Build a weighted graph where edges reflect state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class GraphQNN__gen318:
    """Hybrid graph‑based neural network.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, e.g. ``[n_features, 64, 32, n_outputs]``.
    """

    def __init__(self, arch: Sequence[int]):
        self.arch = list(arch)
        self.weights: List[Tensor] = [
            _random_linear(in_f, out_f) for in_f, out_f in zip(self.arch[:-1], self.arch[1:])
        ]

    def forward(self, x: Tensor) -> Tensor:
        """Compute the network output for a single input vector."""
        current = x
        for weight in self.weights:
            current = torch.tanh(weight @ current)
        return current

    def predict(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Return activations for all samples."""
        return feedforward(self.arch, self.weights, samples)

    def fidelity_graph(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Return the fidelity‑based adjacency graph of the given states."""
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def train_random(self, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate a random training set based on the last layer's weight."""
        target = self.weights[-1]
        return random_training_data(target, samples)


__all__ = [
    "GraphQNN__gen318",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
