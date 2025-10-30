"""GraphQNN: Classical Graph Neural Network with fidelity‑based adjacency training.

This module extends the original seed by adding a small but fully
trainable GNN that operates on the fidelity‑based adjacency graph
produced by the QML counterpart.  The class exposes a `train` method
that returns the best weight matrices and a `predict` function that
produces node‑level outputs.  The implementation uses PyTorch
and networkx for graph utilities."""
from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
EdgeIndex = Tuple[int, int]


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(
    weight: Tensor, samples: int
) -> List[Tuple[Tensor, Tensor]]:
    """Generate (input, target) pairs where target = weight @ input."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random linear network and training data."""
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
    """Apply the linear network to each sample and record activations."""
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
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class GraphQNN__gen123:
    """A lightweight GNN that learns on a fidelity‑based adjacency graph."""
    def __init__(self, arch: Sequence[int], in_features: int = 1):
        """
        Parameters
        ----------
        arch : Sequence[int]
            Hidden layer sizes of the GNN.
        in_features : int, default 1
            Dimensionality of the node feature vector.
        """
        self.arch = list(arch)
        self.in_features = in_features
        layers: List[nn.Module] = []
        prev = in_features
        for h in self.arch:
            layers.append(nn.Linear(prev, h))
            prev = h
        self.layers = nn.ModuleList(layers)
        self.readout = nn.Linear(prev, 1)

    def forward(self, adjacency: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GNN."""
        x = features
        for layer in self.layers:
            x = layer(adjacency @ x)
            x = F.relu(x)
        out = self.readout(adjacency @ x)
        return out

    def train(
        self,
        graph: nx.Graph,
        features: torch.Tensor,
        targets: torch.Tensor,
        epochs: int = 200,
        lr: float = 0.01,
    ) -> "GraphQNN__gen123":
        """Train the GNN on the provided graph and node labels."""
        adj = nx.adjacency_matrix(graph).astype(float)
        adj = torch.tensor(adj.todense(), dtype=torch.float32)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            out = self.forward(adj, features).squeeze()
            loss = loss_fn(out, targets)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, graph: nx.Graph, features: torch.Tensor) -> torch.Tensor:
        """Return node‑level predictions for the given graph."""
        adj = nx.adjacency_matrix(graph).astype(float)
        adj = torch.tensor(adj.todense(), dtype=torch.float32)
        with torch.no_grad():
            return self.forward(adj, features).squeeze()

    def parameters(self) -> List[torch.nn.parameter.Parameter]:
        """Return all learnable parameters."""
        return list(self.layers.parameters()) + list(self.readout.parameters())

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        return state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        return random_network(qnn_arch, samples)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int):
        return random_training_data(weight, samples)

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        return feedforward(qnn_arch, weights, samples)


__all__ = [
    "GraphQNN__gen123",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
