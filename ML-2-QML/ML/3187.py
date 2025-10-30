"""GraphQNNGen119: classical graph neural network + classifier utilities.

The module merges the original GraphQNN functionality with a simple
feed‑forward classifier, all while preserving the quantum‑side API.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import torch
import networkx as nx

Tensor = torch.Tensor


def _rand_lin(in_f: int, out_f: int) -> Tensor:
    """Return a random weight matrix for a linear layer."""
    return torch.randn(out_f, in_f, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a synthetic dataset for supervised learning."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(arch: Sequence[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Create a random feed‑forward network and a training set."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(arch[:-1], arch[1:]):
        weights.append(_rand_lin(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(arch), weights, training_data, target_weight


def feedforward(
    arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Forward propagate a batch of samples through the network."""
    activations: List[List[Tensor]] = []
    for features, _ in samples:
        layer_vals = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            layer_vals.append(current)
        activations.append(layer_vals)
    return activations


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the squared overlap between two vectors."""
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
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(si, sj)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[torch.nn.Module, List[int], List[int], List[int]]:
    """Construct a simple feed‑forward classifier mirroring the quantum design."""
    layers: List[torch.nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []
    for _ in range(depth):
        linear = torch.nn.Linear(in_dim, num_features)
        layers.extend([linear, torch.nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = torch.nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = torch.nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


class GraphQNNGen119:
    """Unified container for the classical graph‑based network and classifier."""

    def __init__(self, arch: Sequence[int], depth: int = 3, samples: int = 100):
        self.arch = list(arch)
        self.depth = depth
        self.weights, self.training_data, self.target_weight = random_network(self.arch, samples)
        self.classifier, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(len(arch[0]), depth)

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Convenience wrapper delegating to the module‑level function."""
        return feedforward(self.arch, self.weights, samples)

    def classify(self, features: Tensor) -> torch.Tensor:
        """Run the classifier on a single feature vector."""
        return self.classifier(features)

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
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        return state_fidelity(a, b)


__all__ = [
    "GraphQNNGen119",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "build_classifier_circuit",
]
