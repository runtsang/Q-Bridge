"""Hybrid classical graph neural network with classifier support.

This module extends the original GraphQNN utilities by adding a
classifier construction routine and a simple hybrid fidelity
computation between the classical network and a quantum counterpart.
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
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

class GraphQNNHybrid:
    """Classical graph‑based neural network with optional classifier."""

    def __init__(self, arch: Sequence[int], depth: int = 2, device: str = "cpu"):
        self.arch = list(arch)
        self.depth = depth
        self.device = device
        self.weights = [ _random_linear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:]) ]
        self._build_classifier()

    @classmethod
    def random_network(cls, arch: Sequence[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Generate a random network, training data and target weight."""
        weights = [ _random_linear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:]) ]
        target_weight = weights[-1]
        training_data = random_training_data(target_weight, samples)
        return list(arch), weights, training_data, target_weight

    def _build_classifier(self) -> None:
        """Instantiate a small feed‑forward classifier."""
        layers: List[nn.Module] = []
        in_dim = self.arch[-1]
        for _ in range(self.depth):
            linear = nn.Linear(in_dim, in_dim)
            layers.append(linear)
            layers.append(nn.ReLU())
            in_dim = in_dim
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        self.classifier = nn.Sequential(*layers).to(self.device)

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Run a forward pass and return layer activations."""
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations: List[Tensor] = [features]
            current = features
            for weight in self.weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared overlap between two normalized tensors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """Create a weighted graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNHybrid.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def classifier_forward(self, x: Tensor) -> Tensor:
        """Forward pass through the classifier."""
        return self.classifier(x)

    def hybrid_fidelity(self, quantum_states: Sequence[Tensor]) -> List[float]:
        """Compute fidelity between classical activations and quantum states."""
        if not quantum_states:
            raise ValueError("No quantum states supplied")
        return [self.state_fidelity(act[-1], qs) for act in self.feedforward(quantum_states)]

__all__ = [
    "GraphQNNHybrid",
    "random_training_data",
]
