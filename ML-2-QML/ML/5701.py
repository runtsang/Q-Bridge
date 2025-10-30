from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn

Tensor = torch.Tensor

@dataclass
class GraphQLayerParameters:
    """Parameters for a single classical graph layer."""
    weight: Tensor
    bias: Tensor

    @staticmethod
    def random(in_features: int, out_features: int) -> "GraphQLayerParameters":
        w = torch.randn(out_features, in_features, dtype=torch.float32)
        b = torch.randn(out_features, dtype=torch.float32)
        return GraphQLayerParameters(w, b)

class GraphQNN(nn.Module):
    """Classical graph neural network with optional weight clipping."""
    def __init__(self, arch: Sequence[int], clip: bool = True):
        super().__init__()
        self.arch = list(arch)
        layers: List[nn.Module] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            linear = nn.Linear(in_f, out_f)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            if clip:
                with torch.no_grad():
                    linear.weight.clamp_(-5.0, 5.0)
                    linear.bias.clamp_(-5.0, 5.0)
            layers.append(linear)
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Return activations per layer for each sample."""
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for layer in self.net:
                current = layer(current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Generate a random network and training data."""
        weights: List[Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(GraphQLayerParameters.random(in_f, out_f).weight)
        target_weight = weights[-1]
        training_data = GraphQNN.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        data = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            data.append((features, target))
        return data

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
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
            fid = GraphQNN.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

__all__ = [
    "GraphQNN",
    "GraphQLayerParameters",
]
