"""Hybrid classical graph neural network that blends CNN feature extraction with
graph‑based feed‑forward propagation.

The implementation borrows the weight‑generation, forward‑propagation and
fidelity‑based graph construction from the original GraphQNN module while
introducing a lightweight CNN front‑end inspired by Quantum‑NAT.
"""

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
        for w in weights:
            current = torch.tanh(w @ current)
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
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class HybridGraphQNN(nn.Module):
    """CNN + classical graph neural network hybrid.

    The network first extracts features with a shallow CNN, then propagates
    those features through a sequence of linear layers whose weights are
    treated as edges in a graph.  The graph structure can be visualised
    via :func:`fidelity_adjacency`.
    """

    def __init__(
        self,
        arch: Sequence[int],
        cnn_channels: Sequence[int] | None = None,
        init_weights: Sequence[Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.arch = list(arch)

        # --- CNN front‑end ---------------------------------------------------
        if cnn_channels is None:
            # Default 2‑layer CNN used in Quantum‑NAT
            cnn_channels = (1, 8, 16)
        layers = []
        in_ch = cnn_channels[0]
        for out_ch in cnn_channels[1:]:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # Flatten size after CNN
        dummy = torch.zeros(1, cnn_channels[0], 28, 28)
        with torch.no_grad():
            out = self.cnn(dummy)
        self.flatten_dim = out.view(1, -1).size(1)

        # --- Graph network ---------------------------------------------------
        if init_weights is None:
            _, self.weights, _, _ = random_network(arch, samples=0)
        else:
            self.weights = list(init_weights)

        self.norm = nn.BatchNorm1d(self.weights[-1].size(0))

    def forward(self, x: Tensor) -> Tensor:
        # CNN feature extraction
        features = self.cnn(x)
        flattened = features.view(x.size(0), -1)
        # Feed‑forward through graph layers
        activations = flattened
        for w in self.weights:
            activations = torch.tanh(w @ activations.t()).t()
        return self.norm(activations)

    def fidelity_graph(self, threshold: float, *, secondary: float | None = None) -> nx.Graph:
        """Build a graph from the activations of the last layer."""
        with torch.no_grad():
            activations = self.forward(torch.randn(1, *self.cnn[0].weight.shape[1:], 28, 28))
        return fidelity_adjacency(activations, threshold, secondary=secondary)

__all__ = [
    "HybridGraphQNN",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
