"""GraphQNNHybrid: classical utilities with a hybrid quanvolution filter.

This module extends the original GraphQNN utilities with a
classical surrogate for the quantum quanvolution filter and a
classifier that can be used on image data.  The public API
mirrors the quantum counterpart so that the same class name can
be imported from either backend.

The design follows a combination scaling paradigm: a purely
classical feed‑forward network is augmented with a quantum‑
inspired filter that can be swapped for a quantum implementation
in the QML module.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
#  Classical feed‑forward utilities
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate (input, target) pairs for a linear layer."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random classical feed‑forward network and a training set."""
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
    """Return a list of activation vectors for each sample."""
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
    """Squared overlap between two classical activation vectors."""
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
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
#  Classical Quanvolution filter
# --------------------------------------------------------------------------- #

class QuanvolutionFilter(nn.Module):
    """A lightweight 2×2 convolution followed by flattening – a
    classical surrogate for the quantum quanvolution filter.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    """Hybrid classifier that uses the classical quanvolution filter
    and a linear output head.
    """
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

# --------------------------------------------------------------------------- #
#  Combined hybrid class
# --------------------------------------------------------------------------- #

class GraphQNNHybrid:
    """Public API that bundles the classical utilities and the
    quanvolution classifier.  The class can be instantiated with a
    GNN architecture and exposes methods for network generation,
    feed‑forward, and graph construction.  It also provides access
    to the QuanvolutionClassifier for image‑based tasks.
    """

    def __init__(self, arch: Sequence[int]) -> None:
        self.arch = list(arch)
        self.weights: List[Tensor] = []
        self.training_data: List[Tuple[Tensor, Tensor]] = []

    def build(self, samples: int = 100) -> None:
        """Generate random weights and a training set for the last layer."""
        _, self.weights, self.training_data, _ = random_network(self.arch, samples)

    def forward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Run a forward pass through the network."""
        return feedforward(self.arch, self.weights, samples)

    def fidelity_graph(self, threshold: float, secondary: float | None = None) -> nx.Graph:
        """Create a graph from the activations of the training data."""
        activations = [acts[-1] for acts in self.forward(self.training_data)]
        return fidelity_adjacency(activations, threshold, secondary=secondary)

    # Expose the classifier as a convenience
    classifier = QuanvolutionClassifier

    __all__ = [
        "GraphQNNHybrid",
        "QuanvolutionFilter",
        "QuanvolutionClassifier",
        "random_network",
        "feedforward",
        "state_fidelity",
        "fidelity_adjacency",
    ]
