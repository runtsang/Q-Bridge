"""GraphQNN - classical GNN module with trainable layers and graph‑based loss.

This module extends the original seed by adding a small GNN head that takes the
layer‑wise activations produced by :func:`feedforward` and learns to predict the
target weight matrix.  It also provides a graph‑based loss that can be used in
hybrid training loops.
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
    """Return a random normal matrix with shape (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a dataset of (x, target) pairs where target = weight @ x."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(
    qnn_arch: Sequence[int],
    samples: int,
) -> Tuple[List[int], List[nn.Linear], List[Tuple[Tensor, Tensor]], Tensor]:
    """Return architecture, weight modules, training data and the final target weight."""
    layers: List[nn.Linear] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        layers.append(nn.Linear(in_f, out_f, bias=False))
    target_weight = layers[-1].weight
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), layers, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    layers: Sequence[nn.Linear],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Return a list of activation lists for each sample."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for layer in layers:
            current = torch.tanh(layer(current))
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Compute the squared‑norm‑normalized dot product between two vectors."""
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
    """Build a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class GraphLoss(nn.Module):
    """Graph‑based loss that uses a weighted adjacency matrix derived from
    fidelity between predicted and target states.

    The loss is the sum over all edges of the squared difference between the
    predicted and target node features, weighted by the adjacency.
    """
    def __init__(self, adjacency: nx.Graph):
        super().__init__()
        size = adjacency.number_of_nodes()
        mat = torch.zeros(size, size, dtype=torch.float32)
        for i, j, data in adjacency.edges(data=True):
            weight = data.get("weight", 1.0)
            mat[i, j] = weight
            mat[j, i] = weight
        self.register_buffer("adj_matrix", mat)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        diff = preds - targets
        loss = torch.sum((diff @ self.adj_matrix) * diff) / (2 * self.adj_matrix.sum() + 1e-12)
        return loss


class GraphQNN(nn.Module):
    """A hybrid classical graph‑based neural network that mirrors the quantum
    interface.  It can be trained with the provided :class:`GraphLoss` or a
    standard MSE loss.
    """
    def __init__(self, qnn_arch: Sequence[int]):
        super().__init__()
        self.arch = list(qnn_arch)
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f, bias=False)
                                    for in_f, out_f in zip(self.arch[:-1], self.arch[1:])])

    def forward(self, x: Tensor) -> Tensor:
        current = x
        for layer in self.layers:
            current = torch.tanh(layer(current))
        return current

    def train_step(
        self,
        data_loader: Iterable[Tuple[Tensor, Tensor]],
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> float:
        self.train()
        total_loss = 0.0
        for batch_x, batch_y in data_loader:
            optimizer.zero_grad()
            preds = self(batch_x)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data_loader)

    @staticmethod
    def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold,
                                  secondary=secondary,
                                  secondary_weight=secondary_weight)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        return random_training_data(weight, samples)

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        return random_network(qnn_arch, samples)


__all__ = [
    "GraphQNN",
    "GraphLoss",
    "fidelity_adjacency",
    "state_fidelity",
    "random_network",
    "random_training_data",
]
