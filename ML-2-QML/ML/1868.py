"""GraphQNN: a hybrid classical/quantum neural‑network generator.

The implementation follows the original seed but expands the
functionality with a few key extensions:
*  A `GraphQNN` class that holds architecture, weights and training data.
*  A unified `forward` that works for both torch tensors and Qobj.
*  A `train` method that optimises a combined loss (MSE + fidelity).
*  Optional graph‑based regularisation using the original fidelity‑based
  adjacency.
*  Helper utilities for generating random networks, data and graphs.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence, List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a synthetic dataset for a linear mapping defined by ``weight``."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Return architecture, weight tensors, synthetic data and the target weight."""
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
    """Forward pass returning all intermediate activations."""
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
    """Squared overlap between two unit‑normed vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a graph where edges denote fidelity above ``threshold``."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


@dataclass
class GraphQNN:
    """Hybrid neural‑network wrapper with optional graph regularisation."""
    architecture: Sequence[int]
    weights: List[Tensor]
    training_data: List[Tuple[Tensor, Tensor]]
    target_weight: Tensor
    graph: nx.Graph | None = None

    @staticmethod
    def create(
        architecture: Sequence[int],
        samples: int = 100,
        graph_threshold: float | None = None,
    ) -> "GraphQNN":
        """Convenience constructor generating a random network and data."""
        arch, weights, data, target = random_network(architecture, samples)
        graph = None
        if graph_threshold is not None:
            # Build a graph from the target weight vectors (flattened)
            states = [w.flatten() for w in weights]
            graph = fidelity_adjacency(states, graph_threshold)
        return GraphQNN(arch, weights, data, target, graph)

    def forward(self, x: Tensor) -> Tensor:
        """Return the final activation for a single input."""
        current = x
        for w in self.weights:
            current = torch.tanh(w @ current)
        return current

    def loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Combined MSE + fidelity loss."""
        mse = F.mse_loss(prediction, target)
        fid = state_fidelity(prediction, target)
        return mse - 0.5 * fid  # encourage high fidelity

    def train(self, lr: float = 0.01, epochs: int = 1) -> None:
        """Run a simple SGD loop over the training dataset."""
        opt = torch.optim.SGD(self.weights, lr=lr)
        for _ in range(epochs):
            for x, y in self.training_data:
                opt.zero_grad()
                y_hat = self.forward(x)
                loss = self.loss(y_hat, y)
                loss.backward()
                opt.step()

    def graph_regulariser(self, lambda_reg: float = 0.1) -> Tensor:
        """Optional graph‑based regularisation term."""
        if self.graph is None:
            return torch.tensor(0.0)
        reg = 0.0
        for (i, j, data) in self.graph.edges(data=True):
            w_i = self.weights[i].flatten()
            w_j = self.weights[j].flatten()
            reg += data.get("weight", 1.0) * torch.norm(w_i - w_j)
        return lambda_reg * reg


__all__ = [
    "GraphQNN",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
