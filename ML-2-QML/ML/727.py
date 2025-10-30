"""Hybrid classical graph neural network with batch support and embedding extraction.

This module extends the original seed by providing a torch.nn.Module that
supports batched input, uses ReLU activations, and returns hidden layer
representations for downstream clustering or graph construction.  It also
includes helper functions to generate random training data, build a random
network, and construct a fidelity‑based graph from the embeddings.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class GraphData:
    """Container for embeddings and optional graph."""
    def __init__(self, embeddings: Tensor, graph: nx.Graph | None = None, **metadata):
        self.embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        self.graph = graph
        self.metadata = metadata


def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[tuple[Tensor, Tensor]]:
    dataset: List[tuple[Tensor, Tensor]] = []
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


class GraphQNN(nn.Module):
    """Simple feed‑forward GNN with ReLU and batch support."""
    def __init__(self, architecture: Sequence[int]):
        super().__init__()
        self.architecture = list(architecture)
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(self.architecture[:-1], self.architecture[1:]):
            self.layers.append(nn.Linear(in_f, out_f))

    def forward(self, x: Tensor) -> list[Tensor]:
        """Return activations for each layer."""
        activations: list[Tensor] = [x]
        for layer in self.layers:
            x = F.tanh(layer(x))
            activations.append(x)
        return activations


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Batch‑aware feed‑forward that returns hidden states."""
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
    """Return squared overlap between two classical vectors."""
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
    """Build weighted adjacency graph from classical state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


__all__ = [
    "GraphQNN",
    "GraphData",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
