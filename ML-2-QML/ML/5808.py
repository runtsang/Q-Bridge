"""
GraphQNNGen201 – Classical hybrid implementation.

This module defines a Torch‑based Graph Neural Network that mirrors the
quantum interface.  It is fully self‑contained, uses only NumPy,
PyTorch and NetworkX, and is compatible with the original seed
GraphQNN.py.  The class exposes methods for random weight generation,
synthetic training data, forward propagation, state fidelity and
fidelity‑based adjacency graph construction.
"""

from __future__ import annotations

import itertools
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import Iterable, List, Sequence, Tuple

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Generate a random weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic (feature, target) pairs for a linear target."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Return architecture, weights, synthetic data and target weight."""
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
    """Return layer‑wise activations for each sample."""
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
    """Squared overlap of two normalized vectors."""
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
    """Build a weighted graph from pairwise fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class GraphQNNGen201(nn.Module):
    """
    Classical implementation of a graph‑based neural network.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, e.g. ``[4, 8, 4]``.
    """

    def __init__(self, arch: Sequence[int]) -> None:
        super().__init__()
        self.arch = list(arch)
        self.layers = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(self.arch[:-1], self.arch[1:])]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Standard feed‑forward with tanh activations."""
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x

    @staticmethod
    def random_network(arch: Sequence[int], samples: int):
        """Convenience wrapper returning architecture, weights, data and target."""
        return random_network(arch, samples)

    @staticmethod
    def feedforward(
        arch: Sequence[int],
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ):
        """Static helper mirroring the original seed function."""
        return feedforward(arch, weights, samples)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ):
        """Static helper mirroring the original seed function."""
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Static wrapper for state_fidelity."""
        return state_fidelity(a, b)

    def random_training_data(self, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate synthetic data for the network's last layer."""
        target_weight = self.layers[-1].weight.data
        return random_training_data(target_weight, samples)

__all__ = [
    "GraphQNNGen201",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
