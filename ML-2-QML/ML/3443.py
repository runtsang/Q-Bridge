"""GraphQNNClassifier: hybrid graph neural network with classical and quantum interfaces.

The module defines a class that exposes both classical feed‑forward and quantum
variational circuits, mirroring the structure of the two seed projects.
All public methods are documented to keep the API compatible with the
original GraphQNN and QuantumClassifierModel utilities.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple, List

import torch
import torch.nn as nn
import networkx as nx
import numpy as np

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Generate a random weight matrix for a linear layer."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training pairs (x, Wx)."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


class GraphQNNClassifier:
    """Hybrid GraphQNN classifier with a classical NN backbone and optional
    quantum circuit interface.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes of the feed‑forward network.
    depth : int
        Depth of the variational ansatz used when the quantum interface
        is enabled.
    """

    def __init__(self, arch: Sequence[int], depth: int):
        self.arch = list(arch)
        self.depth = depth
        self.weights = [_random_linear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:])]
        self.qnn_arch = arch  # reused for the quantum side
        self._classifier = None  # lazy creation of a PyTorch model

    # ------------------------------------------------------------------
    # Classical helpers
    # ------------------------------------------------------------------
    @property
    def classifier(self) -> nn.Module:
        """Return a PyTorch feed‑forward classifier mirroring the quantum
        construction.  The model is built on first use."""
        if self._classifier is None:
            layers: List[nn.Module] = []
            in_dim = self.arch[0]
            for out_dim in self.arch[1:]:
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(nn.ReLU())
                in_dim = out_dim
            layers.append(nn.Linear(in_dim, 2))
            self._classifier = nn.Sequential(*layers)
        return self._classifier

    def feedforward(
        self, samples: Iterable[Tuple[Tensor, Tensor]]
    ) -> List[List[Tensor]]:
        """Classical forward pass returning the activations of all layers."""
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in self.weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared overlap of two feature vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from feature‑vector fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(
            enumerate(states), 2
        ):
            fid = GraphQNNClassifier.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def random_network(
        qnn_arch: Sequence[int], samples: int
    ) -> Tuple[Sequence[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Generate a random classical network, its training data and target."""
        weights: List[Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(_random_linear(in_f, out_f))
        target_weight = weights[-1]
        training_data = random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    # ------------------------------------------------------------------
    # Quantum helpers (stubs for API compatibility)
    # ------------------------------------------------------------------
    def build_quantum_circuit(self) -> None:
        """Placeholder: build a quantum circuit for the classifier.

        In this classical module the method is a no‑op but keeps the API
        compatible with the quantum version.
        """
        pass

    def quantum_feedforward(
        self, samples: Iterable[Tuple[Tensor, Tensor]]
    ) -> List[List[Tensor]]:
        """Placeholder for quantum feed‑forward – returns the same as classical."""
        return self.feedforward(samples)

    def quantum_fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Placeholder for quantum fidelity adjacency – proxies the classical version."""
        return self.fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def quantum_random_network(self, qnn_arch: Sequence[int], samples: int):
        """Placeholder mirroring the quantum random network interface."""
        return self.random_network(qnn_arch, samples)

__all__ = [
    "GraphQNNClassifier",
    "random_training_data",
]
