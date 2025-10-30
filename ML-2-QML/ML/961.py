"""GraphQNN__gen246: Classical GNN with hybrid training interface.

This module extends the original GraphQNN by adding a Graph Convolution
layer that learns a weight per node type, a multi‑label cross‑entropy
loss, and a simple training loop that can be combined with a quantum
sub‑module.  The class exposes a `train` method that accepts a quantum
module instance and runs a joint loss over classical and quantum
predictions.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Any

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

Tensor = torch.Tensor
Graph = nx.Graph

# --------------------------------------------------------------------------- #
# 1.  Utilities copied from the seed
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a synthetic dataset for the last layer."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Return a random weight chain and a training set for the final layer."""
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
    """Classic forward pass that keeps every activation."""
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
    """Return the squared overlap between two classical vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> Graph:
    """Build a graph where edges encode fidelity ≥ threshold."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# 2.  New GNN backbone
# --------------------------------------------------------------------------- #

class GraphConvLayer(nn.Module):
    """Graph Convolution that uses a separate weight matrix for each node type."""

    def __init__(self, in_dim: int, out_dim: int, num_node_types: int):
        super().__init__()
        # weight: shape (num_node_types, out_dim, in_dim)
        self.weight = nn.Parameter(
            torch.randn(num_node_types, out_dim, in_dim, dtype=torch.float32)
        )

    def forward(self, x: Tensor, node_types: Tensor, adjacency: Tensor) -> Tensor:
        """
        x: (N, in_dim) node features
        node_types: (N,) integer type indices
        adjacency: (N, N) adjacency matrix
        """
        # Aggregate neighbor features
        agg = torch.matmul(adjacency, x)  # (N, in_dim)
        # Apply per‑type weight
        out = torch.zeros((x.size(0), self.weight.size(1)), device=x.device)
        for t in range(self.weight.size(0)):
            mask = (node_types == t)
            if mask.any():
                out[mask] = torch.matmul(agg[mask], self.weight[t].t())
        return out


class GraphQNN__gen246(nn.Module):
    """
    Classical Graph Neural Network that supports multi‑label classification
    and can be trained jointly with a quantum module.
    """

    def __init__(
        self,
        arch: Sequence[int],
        num_node_types: int,
        num_classes: int,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.arch = list(arch)
        self.num_node_types = num_node_types
        self.num_classes = num_classes
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        # Build layers
        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(self.arch[:-1], self.arch[1:]):
            layers.append(GraphConvLayer(in_dim, out_dim, self.num_node_types))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

        self.classifier = nn.Linear(self.arch[-1], self.num_classes)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    # --------------------------------------------------------------------- #
    # Forward and utilities
    # --------------------------------------------------------------------- #

    def forward(
        self,
        x: Tensor,
        node_types: Tensor,
        adjacency: Tensor,
    ) -> Tensor:
        """
        Forward pass through the GNN and the final classifier.
        Returns logits of shape (N, num_classes).
        """
        h = x
        for layer in self.layers:
            if isinstance(layer, GraphConvLayer):
                h = layer(h, node_types, adjacency)
            else:
                h = layer(h)
        logits = self.classifier(h)
        return logits

    def forward_from_graph(self, graph: Graph) -> Tensor:
        """
        Convenience wrapper that extracts the required tensors from a
        networkx graph and returns the logits.
        """
        num_nodes = graph.number_of_nodes()
        # Assume all nodes have the same feature dimensionality
        sample_data = next(iter(graph.nodes(data=True)))[1]
        feature_dim = sample_data["feature"].shape[0]
        x = torch.zeros((num_nodes, feature_dim), device=self.device)
        node_types = torch.zeros(num_nodes, dtype=torch.long, device=self.device)
        for i, (_, data) in enumerate(graph.nodes(data=True)):
            x[i] = torch.tensor(data["feature"], device=self.device, dtype=torch.float32)
            node_types[i] = data["type"]
        adjacency = nx.to_numpy_array(graph, dtype=np.float32)
        adjacency = torch.tensor(adjacency, device=self.device)
        return self.forward(x, node_types, adjacency)

    # --------------------------------------------------------------------- #
    # Training utilities
    # --------------------------------------------------------------------- #

    def train_step(self, graph: Graph, labels: Tensor) -> float:
        """
        Perform a single gradient step on one graph.
        """
        self.train()
        logits = self.forward_from_graph(graph)
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(self.device, dtype=torch.float32))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, graph: Graph, labels: Tensor) -> Tensor:
        """
        Return the sigmoid predictions for a graph.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward_from_graph(graph)
            preds = torch.sigmoid(logits)
        return preds.cpu()

    # --------------------------------------------------------------------- #
    # Hybrid training with a quantum module
    # --------------------------------------------------------------------- #

    def train(
        self,
        quantum_module: "GraphQNN__gen246",
        data: List[Tuple[Graph, Tensor]],
        epochs: int = 10,
    ) -> None:
        """
        Jointly train this classical GNN and a quantum module.
        The loss is the sum of the classical BCE loss and the quantum BCE loss.
        """
        self.train()
        quantum_module.train()
        # Combine parameters
        combined_params = list(self.parameters()) + list(quantum_module.parameters())
        optimizer = torch.optim.Adam(combined_params, lr=quantum_module.lr)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for graph, labels in data:
                # Classical predictions
                logits = self.forward_from_graph(graph)
                # Quantum predictions
                input_state = quantum_module.encode_graph_to_state(graph)
                quantum_preds = quantum_module.forward(input_state)
                # Losses
                loss_cls = F.binary_cross_entropy_with_logits(logits, labels.to(self.device))
                loss_qml = F.binary_cross_entropy_with_logits(quantum_preds, labels.to(self.device))
                loss = loss_cls + loss_qml
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs} joint loss: {epoch_loss / len(data):.4f}")

    # --------------------------------------------------------------------- #
    # Static helper methods
    # --------------------------------------------------------------------- #

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Return a random weight chain and a training set for the last layer."""
        return random_network(qnn_arch, samples)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int):
        """Generate a random dataset for the last layer."""
        return random_training_data(weight, samples)

    @staticmethod
    def random_linear(in_features: int, out_features: int) -> Tensor:
        """Return a random linear weight matrix."""
        return _random_linear(in_features, out_features)

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Return the squared overlap between two classical vectors."""
        return state_fidelity(a, b)

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Classic forward pass that keeps every activation."""
        return feedforward(qnn_arch, weights, samples)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> Graph:
        """Build a graph where edges encode fidelity ≥ threshold."""
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)


__all__ = [
    "GraphQNN__gen246",
    "random_network",
    "random_training_data",
    "random_linear",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
