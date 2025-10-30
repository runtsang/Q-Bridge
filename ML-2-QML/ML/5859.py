"""Hybrid classical graph‑sampler neural network.

The module defines a single class `HybridGraphSamplerQNN` that extends
`torch.nn.Module`.  It combines a GNN‑style feed‑forward network,
fidelity‑based graph construction, and an embedded softmax sampler.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def _xavier_linear(in_features: int, out_features: int) -> Tensor:
    """Create a weight matrix with Xavier uniform initialization."""
    weight = torch.empty(out_features, in_features)
    nn.init.xavier_uniform_(weight)
    return weight


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training pairs (x, y) where y = weight @ x."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1))
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Build a random linear network and a matching training set."""
    weights: List[Tensor] = [_xavier_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Return the activation at every layer for each sample."""
    all_activations: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        all_activations.append(activations)
    return all_activations


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two unit‑norm vectors."""
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
    """Build a weighted graph where edges reflect state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class SamplerModule(nn.Module):
    """Simple softmax sampler used inside the hybrid network."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


class HybridGraphSamplerQNN(nn.Module):
    """Hybrid GNN + Sampler neural network."""

    def __init__(self, qnn_arch: Sequence[int], samples: int = 100) -> None:
        super().__init__()
        self.arch, self.weights, self.training_data, self.target = random_network(qnn_arch, samples)
        self.sampler = SamplerModule()

    def forward(self, inputs: Tensor) -> Tensor:
        """Run the GNN feed‑forward and return the sampler output."""
        activations = feedforward(self.arch, self.weights, [(inputs, None)])
        final_state = activations[0][-1]
        return self.sampler(final_state)

    def get_state_graph(self, threshold: float, *, secondary: float | None = None) -> nx.Graph:
        """Construct a fidelity graph from the activations of the last layer."""
        final_states = [act[-1] for act in feedforward(self.arch, self.weights, self.training_data)]
        return fidelity_adjacency(final_states, threshold, secondary=secondary)


__all__ = [
    "HybridGraphSamplerQNN",
    "SamplerModule",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
