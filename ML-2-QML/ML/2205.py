from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple

import torch
import torch.nn as nn
import numpy as np
import networkx as nx

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Generate a random weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Create synthetic training pairs (x, Wx)."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Build a random classical network and training data."""
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
    """Forward pass storing activations at every layer."""
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
    """Cosine‑based fidelity for classical feature vectors."""
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


class GraphQNNHybrid(nn.Module):
    """Hybrid classical CNN + optional quantum module with graph‑based adjacency support."""

    def __init__(
        self,
        cnn_features: nn.Module | None = None,
        out_features: int = 4,
        quantum_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.cnn_features = cnn_features or self._default_cnn()
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, out_features),
        )
        self.norm = nn.BatchNorm1d(out_features)
        self.quantum_module = quantum_module

    @staticmethod
    def _default_cnn() -> nn.Sequential:
        """Default 2‑layer CNN mirroring the Quantum‑NAT encoder."""
        return nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn_features(x)
        flattened = features.view(x.shape[0], -1)
        out = self.fc(flattened)
        out = self.norm(out)
        if self.quantum_module is not None:
            out = self.quantum_module(out)
        return out

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Cosine similarity between two feature vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    def compute_adjacency(
        self,
        features: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Generate a graph from feature similarities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(features)))
        for (i, f_i), (j, f_j) in itertools.combinations(enumerate(features), 2):
            fid = self.state_fidelity(f_i, f_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNNHybrid",
]
