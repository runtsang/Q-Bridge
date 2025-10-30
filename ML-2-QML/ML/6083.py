"""Graph‑based neural network module with optional residual connections and graph regularisation.

The API mirrors the original seed: ``feedforward`` and ``fidelity_adjacency`` are preserved, but the
underlying architecture is now a configurable feed‑forward network that can optionally add a linear
residual from each layer to the next.  A graph‑based regulariser is added to the loss to encourage
similarity between neighbouring nodes in the training graph.  The implementation is fully classical
and uses only NumPy and PyTorch, making it drop‑in for existing scripts that import ``GraphQNN``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence, List, Tuple

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


Tensor = torch.Tensor
State = np.ndarray


@dataclass
class GraphQNNConfig:
    """Configuration for the hybrid residual‑graph neural network."""
    hidden_layers: List[int] = None
    """List of node‑feature dimensions for each hidden layer (first element is input dim)."""
    residual: bool = False
    """Whether to keep a linear residual connection from the output of each layer to the next."""
    graph_reg: float = 0.0
    """Weight of the graph‑based regularisation term added to the loss."""


class GraphQNN:
    """Hybrid residual‑graph neural network."""

    def __init__(self, config: GraphQNNConfig):
        self.config = config
        self.layers: List[nn.Module] = []
        self.residuals: List[nn.Module] = []

        in_dim = config.hidden_layers[0]
        for out_dim in config.hidden_layers[1:]:
            self.layers.append(nn.Linear(in_dim, out_dim))
            if config.residual:
                self.residuals.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

        self.model = nn.Sequential(*self.layers)

    def feedforward(self, x: Tensor) -> List[Tensor]:
        """Return activations for all layers, including the input."""
        activations: List[Tensor] = [x]
        current = x
        for i, layer in enumerate(self.layers):
            current = torch.tanh(layer(current))
            if self.config.residual:
                current = current + self.residuals[i](activations[-1])
            activations.append(current)
        return activations

    def fidelity_adjacency(
        self, states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5
    ) -> nx.Graph:
        """Create a weighted adjacency graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Return the squared absolute overlap between two state vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float((a_norm @ b_norm).item() ** 2)

    def loss_with_graph_reg(
        self, predictions: Tensor, targets: Tensor, graph: nx.Graph
    ) -> Tensor:
        """Mean‑squared error plus a graph‑based regularisation term."""
        mse = F.mse_loss(predictions, targets)
        reg = 0.0
        if self.config.graph_reg > 0 and graph.number_of_edges() > 0:
            for (i, j) in graph.edges():
                reg += torch.norm(predictions[i] - predictions[j]) ** 2
            reg = reg / graph.number_of_edges()
            reg *= self.config.graph_reg
        return mse + reg


def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Create a random linear network and training data."""
    weights: List[Tensor] = []
    for in_features, out_features in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(torch.randn(out_features, in_features))
    target_weight = weights[-1]
    training_data = []
    for _ in range(samples):
        features = torch.randn(target_weight.size(1))
        target = target_weight @ features
        training_data.append((features, target))
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]
) -> List[List[Tensor]]:
    """Legacy wrapper that keeps the original signature."""
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
    "GraphQNN",
    "GraphQNNConfig",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
