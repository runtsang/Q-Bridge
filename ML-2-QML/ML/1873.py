"""GraphQNN__gen058: Classical GNN with batch training and graph‑loss.

The module keeps the original public API (`feedforward`,
`fidelity_adjacency`, `random_network`, `random_training_data`,
`state_fidelity`) but replaces the simple per‑sample loop with a
vectorised batch implementation that can be trained with
PyTorch’s ``optim``.  A lightweight ``GraphLoss`` class is added to
compute a weighted‑graph loss from fidelity‑based similarities.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix with shape ``(out_features, in_features)``."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(
    weight: Tensor,
    samples: int,
) -> List[Tuple[Tensor, Tensor]]:
    """Generate ``samples`` feature/target pairs for the linear model defined by ``weight``."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random linear network, training data and the target weight."""
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
    """Vectorised forward pass for a batch of samples."""
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
    """Return the squared overlap between two vectors."""
    a_norm = a / (a.norm() + 1e-12)
    b_norm = b / (b.norm() + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from pairwise fidelities."""
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
    """Graph‑loss that penalises dissimilarity between states connected in a graph."""
    def __init__(self, adjacency: nx.Graph):
        super().__init__()
        self.adj = adjacency

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # states: (batch, dim)
        normed = states / (states.norm(dim=1, keepdim=True) + 1e-12)
        sim = torch.matmul(normed, normed.t())
        loss = 0.0
        # sum over edges
        for i, j, data in self.adj.edges(data=True):
            weight = data.get("weight", 1.0)
            loss += weight * (1.0 - sim[i, j])
        if self.adj.number_of_edges() == 0:
            return torch.tensor(0.0, dtype=states.dtype, device=states.device)
        return loss / self.adj.number_of_edges()


def train_batch(
    arch: Sequence[int],
    weights: Sequence[Tensor],
    training_data: List[Tuple[Tensor, Tensor]],
    adjacency: nx.Graph,
    epochs: int = 200,
    lr: float = 1e-3,
) -> List[Tensor]:
    """Train the network to minimise the graph‑loss using Adam."""
    params = [w.clone().detach().requires_grad_(True) for w in weights]
    optimizer = optim.Adam(params, lr=lr)
    loss_fn = GraphLoss(adjacency)

    for _ in range(epochs):
        optimizer.zero_grad()
        # collect predictions
        preds = []
        for feat, _ in training_data:
            cur = feat
            for p in params:
                cur = torch.tanh(p @ cur)
            preds.append(cur)
        preds = torch.stack(preds)
        loss = loss_fn(preds)
        loss.backward()
        optimizer.step()
    return [p.detach() for p in params]


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphLoss",
    "train_batch",
]
