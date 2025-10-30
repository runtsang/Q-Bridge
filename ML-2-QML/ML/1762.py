"""Hybrid Graph Neural Network (classical side).

This module extends the original seed by adding a trainable linear layer that
maps the last hidden activation to a target weight.  A graph‑based regulariser
is added to the loss to encourage smoothness between neighbouring nodes
identified via state fidelities.
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
    """Return a weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate `samples` pairs (x, y) with y = weight @ x."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(
    qnn_arch: Sequence[int], samples: int
) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Return an architecture, randomly initialised weights, training data
    and the target weight that the model will try to learn.
    """
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
    """Return activations for each sample."""
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
    """Cosine similarity squared."""
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
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity greater than or equal to ``threshold`` receive weight 1.
    When ``secondary`` is provided, fidelities between ``secondary`` and
    ``threshold`` are added with ``secondary_weight``.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class GraphQNN__gen351(nn.Module):
    """Hybrid classical Graph QNN with graph‑regularised training."""

    def __init__(
        self,
        arch: Sequence[int],
        learning_rate: float = 1e-3,
        graph_threshold: float = 0.9,
        secondary_threshold: float | None = None,
        secondary_weight: float = 0.5,
    ):
        super().__init__()
        self.arch = list(arch)
        self.graph_threshold = graph_threshold
        self.secondary_threshold = secondary_threshold
        self.secondary_weight = secondary_weight

        # initialise linear layers
        self.layers: nn.ModuleList = nn.ModuleList()
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x: Tensor) -> Tensor:
        """Return the final activation."""
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x

    def _build_graph(self, activations: List[Tensor]) -> nx.Graph:
        """Build a graph from the last hidden layer activations."""
        return fidelity_adjacency(
            activations,
            self.graph_threshold,
            secondary=self.secondary_threshold,
            secondary_weight=self.secondary_weight,
        )

    def train_step(self, batch: List[Tuple[Tensor, Tensor]]) -> torch.Tensor:
        """Perform one gradient step and return the loss."""
        self.train()
        self.optimizer.zero_grad()

        # compute activations for each sample
        activations_list: List[List[Tensor]] = []
        for x, _ in batch:
            x = x
            activations = [x]
            for layer in self.layers:
                x = torch.tanh(layer(x))
                activations.append(x)
            activations_list.append(activations)

        # use last hidden layer activations for graph regulariser
        last_layer = [acts[-2] for acts in activations_list]  # before final output
        graph = self._build_graph(last_layer)

        # loss = MSE + graph regulariser
        mse_loss = 0.0
        for (x, target), acts in zip(batch, activations_list):
            output = acts[-1]
            mse_loss += F.mse_loss(output, target, reduction="sum")
        mse_loss /= len(batch)

        reg_loss = 0.0
        for i, j, data in graph.edges(data=True):
            weight = data["weight"]
            reg_loss += weight * torch.norm(last_layer[i] - last_layer[j]) ** 2
        reg_loss /= len(batch)

        loss = mse_loss + reg_loss
        loss.backward()
        self.optimizer.step()
        return loss

    def fit(self, dataset: List[Tuple[Tensor, Tensor]], epochs: int = 100) -> List[float]:
        """Simple training loop."""
        losses: List[float] = []
        for epoch in range(epochs):
            loss = self.train_step(dataset)
            losses.append(loss.item())
        return losses

    def predict(self, x: Tensor) -> Tensor:
        return self.forward(x)


__all__ = [
    "GraphQNN__gen351",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
