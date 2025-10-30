"""Graph‑Quantum Neural Network – Classical implementation.

This module combines the classical GNN utilities from `GraphQNN.py` with
a simple sampler network.  All functions are fully compatible with the
anchor `GraphQNN.py` interface while adding a `SamplerQNN` component.
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
    for in_features, out_features in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_features, out_features))
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
        for weight in weights:
            current = torch.tanh(weight @ current)
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
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#  Sampler network – a lightweight torch implementation
# --------------------------------------------------------------------------- #
def SamplerQNN() -> nn.Module:
    """Return a small softmax sampler network."""
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
            return F.softmax(self.net(inputs), dim=-1)

    return SamplerModule()


# --------------------------------------------------------------------------- #
#  Hybrid Graph‑QNN class
# --------------------------------------------------------------------------- #
class GraphQNNGen096:
    """Hybrid Graph‑Quantum Neural Network (classical side)."""

    def __init__(self, qnn_arch: Sequence[int], samples: int = 32):
        self.arch, self.weights, self.training_data, self.target_weight = random_network(
            list(qnn_arch), samples
        )
        self.sampler = SamplerQNN()

    def forward(self, inputs: Tensor) -> Tensor:
        """Run the feed‑forward network and sampler."""
        activations = feedforward(self.arch, self.weights, [(inputs, None)])
        final_state = activations[0][-1]
        return self.sampler(final_state)

    def fidelity_graph(self, threshold: float, secondary: float | None = None) -> nx.Graph:
        """Build a graph from state fidelities of the training set."""
        states = [batch[0] for batch in self.training_data]
        return fidelity_adjacency(states, threshold, secondary=secondary)

__all__ = [
    "GraphQNNGen096",
    "SamplerQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
