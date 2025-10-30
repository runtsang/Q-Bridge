"""Hybrid QCNN model with graph‑based data support for classical training.

This module exposes `QCNNGraphHybrid`, a PyTorch model that mirrors the
original QCNN architecture and can be initialised from a fidelity
graph.  It also provides utilities from the original GraphQNN
module to generate random training data and to construct a
similarity graph from feature vectors.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import List, Tuple

import networkx as nx
import torch
from torch import nn

Tensor = torch.Tensor


class QCNNGraphHybrid(nn.Module):
    """A classical neural network that mimics the QCNN architecture.

    Parameters
    ----------
    init_from_graph : nx.Graph | None
        When supplied, the graph is used to bias the initial weight
        matrices.  Edges of weight 1 shift the mean towards identity,
        edges of weight 0.5 towards zero, all other edges use a random
        normal distribution.
    seed : int | None
        Random seed for reproducible initialization.
    """

    def __init__(self, init_from_graph: nx.Graph | None = None, seed: int | None = None) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        # Feature‑map layer
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())

        # Convolution‑like layers
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

        if init_from_graph is not None:
            self._initialize_from_graph(init_from_graph)

    def _initialize_from_graph(self, graph: nx.Graph) -> None:
        """Bias the initial weight matrices from a graph adjacency."""
        layers = [
            self.feature_map,
            self.conv1,
            self.pool1,
            self.conv2,
            self.pool2,
            self.conv3,
            self.head,
        ]
        for layer in layers:
            w_shape = layer[0].weight.shape
            init = torch.randn(w_shape)
            # Simple bias: average edge weight
            if graph.number_of_edges() > 0:
                bias = sum(data.get("weight", 0.0) for _, _, data in graph.edges(data=True))
                bias /= graph.number_of_edges()
                init += bias
            layer[0].weight.data = init
            if layer[0].bias is not None:
                layer[0].bias.data.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

    @staticmethod
    def random_network_from_graph(
        graph: nx.Graph,
        samples: int = 100,
        seed: int | None = None,
    ) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """
        Generate a random linear network and a training set where the target
        is produced by the last layer weight matrix.  The graph is only used
        to bias the weight initialization.
        """
        if seed is not None:
            torch.manual_seed(seed)

        arch = [8, 16, 16, 12, 8, 4, 4, 1]
        weights = [torch.randn(out, in_) for in_, out in zip(arch[:-1], arch[1:])]

        target_weight = weights[-1]
        training_data = []
        for _ in range(samples):
            features = torch.randn(arch[0])
            target = target_weight @ features
            training_data.append((features, target))

        return arch, weights, training_data, target_weight

    @staticmethod
    def feedforward(
        qnn_arch: List[int],
        weights: List[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Execute a forward pass through a sequence of linear layers."""
        activations = []
        for features, _ in samples:
            layer_acts = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                layer_acts.append(current)
            activations.append(layer_acts)
        return activations

    @staticmethod
    def fidelity_adjacency(
        states: List[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Construct a weighted graph from the squared dot product of
        normalised feature vectors."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for i, vi in enumerate(states):
            for j, vj in enumerate(states[i + 1 :], start=i + 1):
                fid = torch.dot(
                    vi / (torch.norm(vi) + 1e-12),
                    vj / (torch.norm(vj) + 1e-12),
                ).item() ** 2
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph


__all__ = [
    "QCNNGraphHybrid",
    "random_network_from_graph",
    "feedforward",
    "fidelity_adjacency",
]
