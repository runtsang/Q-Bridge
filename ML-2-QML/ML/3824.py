"""GraphQNNHybrid: classical GNN with optional quanvolution preprocessing.

This module extends the original GraphQNN utilities with a hybrid
architecture that can incorporate a classical quanvolution filter.
It retains compatibility with the original seed functions while
providing a reusable class for downstream experiments.
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
# Classical quanvolution filter
# --------------------------------------------------------------------------- #
class ClassicalQuanvolutionFilter(nn.Module):
    """Simple 2×2 convolutional filter that mimics the behaviour of the
    original quanvolution example.  It is used only when the user
    explicitly requests quanvolution preprocessing.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        # Expect input of shape (batch, height, width) or (batch, 1, 2, 2).
        features = self.conv(x)
        return features.view(x.size(0), -1)

# --------------------------------------------------------------------------- #
# Core hybrid class
# --------------------------------------------------------------------------- #
class GraphQNNHybrid(nn.Module):
    """Hybrid graph neural network that can optionally prepend a
    classical quanvolution filter before the linear layers.
    """
    def __init__(self, qnn_arch: Sequence[int], use_quanvolution: bool = False) -> None:
        super().__init__()
        self.arch = list(qnn_arch)
        self.use_quanvolution = use_quanvolution
        self.linears = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
        )
        self.quanvolution = ClassicalQuanvolutionFilter() if use_quanvolution else None

    def forward(self, x: Tensor) -> Tensor:
        if self.quanvolution is not None:
            # Reshape to a 2×2 image patch for the filter (requires 4 features)
            x = x.view(x.size(0), 1, 2, 2)
            x = self.quanvolution(x)
        for linear in self.linears:
            x = torch.tanh(linear(x))
        return x

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Run a forward pass through the network and collect all activations."""
        stored: List[List[Tensor]] = []
        for inp, _ in samples:
            activations = [inp]
            current = inp
            if self.quanvolution is not None:
                current = self.quanvolution(current.view(current.size(0), 1, 2, 2))
            for linear in self.linears:
                current = torch.tanh(linear(current))
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Cosine similarity between two feature vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                           secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNHybrid.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1))
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

__all__ = [
    "GraphQNNHybrid",
    "random_network",
    "random_training_data",
    "feedforward",
]
