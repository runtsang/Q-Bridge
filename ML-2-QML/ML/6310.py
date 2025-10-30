"""Hybrid classical graph neural network with integrated quanvolution features."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training pairs for a linear target."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Construct a random classical graph network and training data."""
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
    """Forward‑pass for the classical graph network."""
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
    """Squared overlap between two classical feature vectors."""
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


class QuanvolutionFilter(nn.Module):
    """Classical approximation of the quantum patch encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


class GraphQNNQuanvolution:
    """Classical graph neural network layer that can be used standalone or inside a hybrid."""

    def __init__(self, arch: Sequence[int]) -> None:
        self.arch = list(arch)
        self.weights = [
            torch.randn(out, in_f, dtype=torch.float32)
            for in_f, out in zip(arch[:-1], arch[1:])
        ]

    def __call__(self, features: Tensor) -> Tensor:
        current = features
        for weight in self.weights:
            current = torch.tanh(weight @ current)
        return current


class QuanvolutionClassifier(nn.Module):
    """Hybrid classifier combining quanvolution and a linear head."""

    def __init__(self, arch: Sequence[int]) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.graph = GraphQNNQuanvolution(arch)
        self.linear = nn.Linear(4 * 14 * 14 + arch[-1], 10)

    def forward(self, x: Tensor) -> Tensor:
        features = self.qfilter(x)
        graph_out = self.graph(features)
        combined = torch.cat([features, graph_out], dim=1)
        logits = self.linear(combined)
        return F.log_softmax(logits, dim=-1)


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "QuanvolutionFilter",
    "GraphQNNQuanvolution",
    "QuanvolutionClassifier",
]
