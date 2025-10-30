"""
GraphQNN__gen335.ml
A fully PyTorch‑based implementation that extends the original GraphQNN utilities.
It introduces a learnable aggregation layer, a simple MSE training loop,
and helper functions for generating synthetic datasets and fidelity graphs.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training data using the target weight matrix."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random classical network and a matching training set."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
    """Forward‑propagate a batch of samples through the network."""
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
    """Squared overlap between two classical feature vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph from pairwise state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# ---------------------------------------------------------------------------

class GraphQNNML(nn.Module):
    """
    A hybrid classical network that mirrors the QNN interface but adds
    a learnable aggregation layer and a simple MSE training routine.
    """

    def __init__(self, arch: Sequence[int]):
        super().__init__()
        self.arch = list(arch)
        self.layers = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(self.arch[:-1], self.arch[1:])]
        )
        # Aggregation layer that projects the last hidden state to the output space
        self.agg = nn.Linear(self.arch[-1], self.arch[-1])

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for layer in self.layers:
            h = torch.tanh(layer(h))
        return self.agg(h)

    def train_network(self, data: List[Tuple[Tensor, Tensor]], lr: float = 1e-3, epochs: int = 200):
        """Train the network using MSE loss."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        for _ in range(epochs):
            for x, y in data:
                optimizer.zero_grad()
                pred = self.forward(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
        return self

    def embed(self, x: Tensor) -> Tensor:
        """Return the embedding from the last hidden layer."""
        h = x
        for layer in self.layers:
            h = torch.tanh(layer(h))
        return h

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNNML",
]
