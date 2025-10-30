"""Combined classical Graph Neural Network utilities and classifier.

This module merges the functionality of the original GraphQNN and
QuantumClassifierModel seeds.  It provides a GraphQNN class that
offers static methods for network generation,
forward propagation, fidelity based graph construction, and a
classical feed‑forward classifier builder.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Tuple, Sequence

import networkx as nx
import torch
import torch.nn as nn


Tensor = torch.Tensor


class GraphQNN:
    """Classical graph‑based neural network utilities."""

    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        """Return a random weight matrix."""
        return torch.randn(out_features, in_features, dtype=torch.float32)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate synthetic (x, y) pairs where y = Wx."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Create a random feed‑forward network compatible with the GraphQNN API."""
        weights: List[Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(GraphQNN._random_linear(in_f, out_f))
        target_weight = weights[-1]
        training_data = GraphQNN.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Propagate each sample through the network, storing intermediate activations."""
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared overlap between two classical feature vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
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
            fid = GraphQNN.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def build_classifier_circuit(
        num_features: int, depth: int
    ) -> Tuple[nn.Module, List[int], List[int], list[int]]:
        """Return a classical feed‑forward classifier mirroring the Qiskit helper."""
        layers: List[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: List[int] = []
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.extend([linear, nn.ReLU()])
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        network = nn.Sequential(*layers)
        observables = list(range(2))
        return network, encoding, weight_sizes, observables


__all__ = [
    "GraphQNN",
]
