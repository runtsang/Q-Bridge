"""Graph‑based classical neural network with QCNN‑style layer masking."""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple, List

import networkx as nx
import torch
import torch.nn as nn

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix with shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic (input, target) pairs for a linear target."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(
    qnn_arch: Sequence[int], samples: int, threshold: float = 0.8
) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Create a random weight matrix chain and a training set for the last layer."""
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
    """Propagate inputs through the network, storing activations."""
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
    """Normalized squared dot product between two vectors."""
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


class GraphQNNModel(nn.Module):
    """
    A graph‑aware GNN that masks weight matrices using an adjacency matrix.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes.
    adjacency : nx.Graph
        Graph whose edges dictate which weights are retained.
    """

    def __init__(self, arch: Sequence[int], adjacency: nx.Graph) -> None:
        super().__init__()
        self.arch = list(arch)
        self.adj = adjacency
        self.layers: nn.ModuleList = nn.ModuleList()
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            weight = nn.Parameter(_random_linear(in_f, out_f))
            mask = torch.ones_like(weight)
            # zero out connections where no edge exists
            for i in range(in_f):
                for j in range(out_f):
                    if not adjacency.has_edge(i, j):
                        mask[i, j] = 0.0
            weight.data *= mask
            self.layers.append(weight)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = torch.tanh(layer @ x)
        return x


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNNModel",
]
