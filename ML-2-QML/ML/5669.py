"""GraphQNN__gen047.py – Classical side of the hybrid pipeline.

The class :class:`GraphQNNGen047` implements a lightweight feed‑forward network that can be trained on the same data
used by the quantum version.  It keeps the original ``random_network`` helper
to generate data, but adds a ``train_one_epoch`` method that performs a single
epoch of Adam optimisation.  The implementation is intentionally minimal – it
is meant to be a drop‑in replacement for the original ``GraphQNN.py`` in a
research notebook.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
from torch import nn

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Helper functions – unchanged from the seed
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(
    weight: Tensor, samples: int
) -> List[Tuple[Tensor, Tensor]]:
    """Generate a toy dataset that mirrors the target weight matrix."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(
    qnn_arch: Sequence[int], samples: int
) -> tuple[list[int], list[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Return all objects used by the original seed – weights, training data and
    the target weight that we will try to learn."""
    weights: List[Tensor] = []
    for in_, out in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_, out))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Forward‑pass through the tanh‑activated MLP."""
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
    """Compute the squared overlap between two classical vectors."""
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
    """Create a weighted graph from pairwise fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(
        enumerate(states), 2
    ):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# Main class – an MLP that can be trained
# --------------------------------------------------------------------------- #
class GraphQNNGen047(nn.Module):
    """A lightweight feed‑forward network that can be trained on the same data
    used by the quantum version.  It exposes a ``train_one_epoch`` method
    that performs a single epoch of Adam optimisation.
    """

    def __init__(self, arch: Sequence[int], device: str | torch.device = "cpu"):
        super().__init__()
        self.arch = list(arch)
        self.device = torch.device(device)
        self.layers = nn.ModuleList()
        for in_, out in zip(arch[:-1], arch[1:]):
            layer = nn.Linear(in_, out, bias=False)
            nn.init.normal_(layer.weight)
            self.layers.append(layer)
        self.to(self.device)

    @staticmethod
    def random_network(arch: Sequence[int], samples: int):
        """Generate a network and a training dataset."""
        arch, weights, training_data, target_weight = random_network(arch, samples)
        return arch, weights, training_data, target_weight

    @staticmethod
    def random_training_data(weight: Tensor, samples: int):
        """Generate a toy dataset that mirrors the target weight matrix."""
        return random_training_data(weight, samples)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network."""
        current = x
        for layer in self.layers:
            current = torch.tanh(layer(current))
        return current

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Return activations for each sample."""
        return feedforward(self.arch, [l.weight for l in self.layers], samples)

    def train_one_epoch(self, samples: Iterable[Tuple[Tensor, Tensor]], lr: float = 1e-3):
        """Run a single epoch of Adam optimisation."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.train()
        running_loss = 0.0
        for features, target in samples:
            optimizer.zero_grad()
            output = self.forward(features.to(self.device))
            loss = loss_fn(output, target.to(self.device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(samples)

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Compute the squared overlap between two classical vectors."""
        return state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """Create a weighted graph from pairwise fidelities."""
        return fidelity_adjacency(states, threshold, secondary=secondary,
                                  secondary_weight=secondary_weight)


__all__ = [
    "GraphQNNGen047",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
