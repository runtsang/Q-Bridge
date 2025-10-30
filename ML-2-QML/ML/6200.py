"""
GraphQNN – classical side of a hybrid graph‑based neural network.

Features added:
* MLP class mirroring the original architecture.
* Random data generation for linear targets.
* Layer‑wise fidelity regulariser and graph construction.
* Utility to build an adjacency graph from activations.
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

# --------------------------------------------------------------------------- #
# 1.  MLP – a lightweight feed‑forward network
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """Simple feed‑forward network that mirrors the qnn_arch sizes."""
    def __init__(self, qnn_arch: Sequence[int]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
        )
        self.act = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = self.act(layer(x))
        return x

# --------------------------------------------------------------------------- #
# 2.  Random data utilities
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a weight matrix with the same shape as a torch linear layer."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(
    weight: Tensor,
    samples: int,
    *,
    noise: float = 0.0,
) -> List[Tuple[Tensor, Tensor]]:
    """Generate a synthetic dataset where the target is the linear transform
    given by ``weight``. ``noise`` adds Gaussian noise to the target."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        feature = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ feature
        if noise > 0.0:
            target += torch.randn_like(target) * noise
        dataset.append((feature, target))
    return dataset

def random_network(
    qnn_arch: Sequence[int],
    samples: int,
) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Build architecture, weights, training data, and target weight."""
    weights: List[Tensor] = [
        _random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])
    ]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

# --------------------------------------------------------------------------- #
# 3.  Feedforward
# --------------------------------------------------------------------------- #
def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Return layerwise activations for each sample."""
    stored: List[List[Tensor]] = []
    for feature, _ in samples:
        activations: List[Tensor] = [feature]
        current = feature
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

# --------------------------------------------------------------------------- #
# 4.  Fidelity utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the squared overlap of two normalized vectors."""
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
    """Create a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(si, sj)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# 5.  Graph from activations
# --------------------------------------------------------------------------- #
def activations_graph(
    activations: List[List[Tensor]],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a graph from the last‑layer activations of a batch."""
    last_layer = [act[-1] for act in activations]
    return fidelity_adjacency(last_layer, threshold, secondary=secondary, secondary_weight=secondary_weight)

__all__ = [
    "MLP",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "activations_graph",
]
