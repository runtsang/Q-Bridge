"""Hybrid classical graph‑quantum neural network utilities.

This module extends the original GraphQNN with convolution‑style
fully‑connected blocks inspired by QCNN.  The resulting
`HybridGraphQCNN` class can be trained with standard PyTorch
pipelines and exposes helper functions for random data generation
and fidelity‑based graph construction."""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two vectors."""
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
    """Create a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(
        enumerate(states), 2
    ):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic (x, y) pairs for a linear map."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Return architecture, random weights, training data and target."""
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
    """Propagate a batch of samples through the linear graph layers."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


class HybridGraphQCNN(nn.Module):
    """Hybrid classical model combining graph layers and QCNN‑style blocks."""

    def __init__(self, arch: Sequence[int], conv_layers: int = 3) -> None:
        super().__init__()
        self.arch = list(arch)
        self.conv_layers = conv_layers

        # Graph linear layers
        self.graph_layers = nn.ModuleList(
            nn.Linear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:])
        )

        # Convolutional refinement blocks
        self.conv_blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(arch[-1], arch[-1]), nn.Tanh())
            for _ in range(conv_layers)
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.graph_layers:
            x = torch.tanh(layer(x))
        for conv in self.conv_blocks:
            x = conv(x)
        return x

    @staticmethod
    def random_model(
        arch: Sequence[int],
        samples: int = 1000,
        conv_layers: int = 3,
    ) -> tuple["HybridGraphQCNN", List[Tuple[Tensor, Tensor]]]:
        """Convenience constructor that returns a ready‑to‑train model
        and synthetic training data."""
        _, weights, training_data, _ = random_network(arch, samples)
        model = HybridGraphQCNN(arch, conv_layers)
        # initialise weights
        with torch.no_grad():
            for layer, w in zip(model.graph_layers, weights):
                layer.weight.copy_(w)
        return model, training_data


__all__ = [
    "HybridGraphQCNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
