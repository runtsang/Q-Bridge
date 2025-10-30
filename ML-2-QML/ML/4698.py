"""
GraphQNNHybrid – Classical implementation

Provides:
- Graph‑based neural network with random weighted layers
- Fidelity‑based adjacency construction
- RBF kernel matrix
- Optional QCNN head for classification
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
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


def random_network(qnn_arch: Sequence[int], samples: int):
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
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class KernalAnsatz(nn.Module):
    """Radial basis function kernel ansatz."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Wrapper for :class:`KernalAnsatz`."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: Sequence[Tensor], b: Sequence[Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


class QCNNModel(nn.Module):
    """Classical surrogate of a quantum convolutional neural network."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNModel:
    """Factory returning the configured :class:`QCNNModel`."""
    return QCNNModel()


class GraphQNNHybrid:
    """
    Hybrid Graph Neural Network with optional QCNN head.

    Public API mirrors the original GraphQNN module while adding a
    quantum‑kernel similarity measure and a QCNN classifier head.
    """

    def __init__(self, architecture: Sequence[int], gamma: float = 1.0, seed: int | None = None) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        self.architecture = list(architecture)
        self.gamma = gamma
        self.weights: List[Tensor] | None = None
        self.training_data: List[Tuple[Tensor, Tensor]] | None = None
        self.qcnn_head = QCNN()

    def initialize(self, samples: int = 32) -> None:
        """Create random weights and training data for the network."""
        _, self.weights, self.training_data, _ = random_network(self.architecture, samples)

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Forward pass through the classical network."""
        if self.weights is None:
            raise RuntimeError("Weights not initialized – call `initialize()` first.")
        return feedforward(self.architecture, self.weights, samples)

    def fidelity_graph(self, activations: Sequence[Tensor], threshold: float) -> nx.Graph:
        """Build a graph where nodes are activations and edges reflect fidelity."""
        return fidelity_adjacency(activations, threshold)

    def kernel(self, a: Sequence[Tensor], b: Sequence[Tensor]) -> np.ndarray:
        """Compute an RBF kernel matrix between two sets of activations."""
        return kernel_matrix(a, b, self.gamma)

    def classify_from_network(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> Tensor:
        """
        Apply the QCNN head to the last‑layer activation of each sample.
        Returns a tensor of classification probabilities.
        """
        activations = self.feedforward(samples)
        last_layer = [a[-1] for a in activations]
        last_tensor = torch.stack(last_layer)
        return self.qcnn_head(last_tensor)


__all__ = [
    "GraphQNNHybrid",
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "QCNNModel",
    "QCNN",
]
