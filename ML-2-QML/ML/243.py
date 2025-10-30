"""Hybrid classical Graph Neural Network utilities.

This module extends the original GraphQNN utilities by wrapping the
feedforward, fidelity, and random data generation into a lightweight
PyTorch class.  It also adds a simple Message‑Passing Graph Neural
Network that can be trained on the fidelity adjacency graph produced
by the quantum circuit, allowing a side‑by‑side classical comparison.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

Tensor = torch.Tensor


class GraphQNN:
    """A simple classical graph neural network wrapper."""
    def __init__(self, arch: Sequence[int], device: torch.device | str = "cpu"):
        self.arch = list(arch)
        self.device = torch.device(device)
        self.weights: List[Tensor] = [
            torch.randn(out, in_, dtype=torch.float32, device=self.device)
            for in_, out in zip(self.arch[:-1], self.arch[1:])
        ]

    # --------------------------------------------------------------------- #
    # Basic utilities (unchanged from the seed)
    # --------------------------------------------------------------------- #
    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        return torch.randn(out_features, in_features, dtype=torch.float32)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(arch: Sequence[int], samples: int) -> Tuple["GraphQNN", List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Return a GraphQNN instance, its weights, training data and the target weight."""
        weights: List[Tensor] = [
            GraphQNN._random_linear(in_, out_) for in_, out_ in zip(arch[:-1], arch[1:])
        ]
        target_weight = weights[-1]
        training_data = GraphQNN.random_training_data(target_weight, samples)
        instance = GraphQNN(arch)
        instance.weights = weights
        return instance, weights, training_data, target_weight

    # --------------------------------------------------------------------- #
    # Forward propagation
    # --------------------------------------------------------------------- #
    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations: List[Tensor] = [features]
            current = features
            for weight in self.weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    # --------------------------------------------------------------------- #
    # Fidelity utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # --------------------------------------------------------------------- #
    # Simple graph neural network for node embeddings
    # --------------------------------------------------------------------- #
    class _GNN(MessagePassing):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__(aggr="add")
            self.lin = nn.Linear(in_channels, out_channels)

        def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
            return self.propagate(edge_index, x=x)

        def message(self, x_j: Tensor) -> Tensor:
            return F.relu(self.lin(x_j))

    def node_embeddings(self, graph: nx.Graph, hidden_dim: int = 16) -> Tensor:
        """Return a node embedding matrix for the given graph."""
        edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
        num_nodes = graph.number_of_nodes()
        x = torch.randn(num_nodes, hidden_dim, device=self.device)
        gnn = self._GNN(hidden_dim, hidden_dim).to(self.device)
        return gnn(x, edge_index)

    # --------------------------------------------------------------------- #
    # Training utilities (simple linear layer training)
    # --------------------------------------------------------------------- #
    def train(self, dataset: List[Tuple[Tensor, Tensor]], epochs: int = 10, lr: float = 1e-3) -> None:
        optimizer = torch.optim.Adam(self.weights, lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            for features, target in dataset:
                features = features.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                out = features
                for weight in self.weights:
                    out = torch.tanh(weight @ out)
                loss = loss_fn(out, target)
                loss.backward()
                optimizer.step()

    def evaluate(self, dataset: List[Tuple[Tensor, Tensor]]) -> float:
        loss_fn = nn.MSELoss()
        total = 0.0
        for features, target in dataset:
            features = features.to(self.device)
            target = target.to(self.device)
            out = features
            for weight in self.weights:
                out = torch.tanh(weight @ out)
            total += loss_fn(out, target).item()
        return total / len(dataset)


__all__ = [
    "GraphQNN",
]
