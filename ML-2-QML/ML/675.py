"""Hybrid graph‑based neural network utilities with a classical optimisation loop.

The module builds upon the original random‑network generator and
fidelity‑based adjacency construction, but now exposes a full training
pipeline that samples a batch of inputs, propagates them through a
classical feed‑forward network, and computes a graph‑based loss that
can be used to pre‑train the classical part before a quantum fine‑tuning
step.

The code is fully importable and uses only NumPy, PyTorch and
networkx.  It is deliberately kept lightweight so it can be dropped
into a research notebook or a larger pipeline.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import numpy as np

Tensor = torch.Tensor
Array = np.ndarray


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(
    weight: Tensor,
    samples: int,
) -> List[Tuple[Tensor, Tensor]]:
    """Generate a batch of (x, y) pairs where y = Wx."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        x = torch.randn(weight.size(1), dtype=torch.float32)
        y = weight @ x
        dataset.append((x, y))
    return dataset


def random_network(
    qnn_arch: Sequence[int],
    samples: int,
) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Create a random network and its training data.

    The output is a list of layer sizes, *and* a list of weight matrices
    that will **be‑told** to the set‑point target weight.
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
    """Propagate each sample through the network and collect activations."""
    stored: List[List[Tensor]] = []
    for x, _ in samples:
        activations = [x]
        current = x
        for w in weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the squared overlap between two unit vectors."""
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
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class GraphQNN:
    """Hybrid graph‑based neural network with classical training support."""

    def __init__(self, architecture: Sequence[int]):
        self.architecture = list(architecture)
        self.weights: List[Tensor] = [
            _random_linear(in_f, out_f)
            for in_f, out_f in zip(self.architecture[:-1], self.architecture[1:])
        ]
        self.target_weight: Tensor | None = None

    def set_target(self, target: Tensor) -> None:
        """Store a target weight matrix for supervised training."""
        self.target_weight = target

    def forward(self, x: Tensor) -> Tensor:
        """Single‑sample forward pass."""
        current = x
        for w in self.weights:
            current = torch.tanh(w @ current)
        return current

    def train(
        self,
        data: Iterable[Tuple[Tensor, Tensor]],
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> None:
        """Train the network to match the target weight using MSE loss."""
        if self.target_weight is None:
            raise ValueError("Target weight not set. Use `set_target` before training.")
        optim = torch.optim.Adam(self.weights, lr=lr)
        loss_fn = torch.nn.MSELoss()
        for _ in range(epochs):
            for x, y in data:
                optim.zero_grad()
                pred = self.forward(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optim.step()

    def graph_loss(
        self,
        samples: Iterable[Tuple[Tensor, Tensor]],
        threshold: float,
    ) -> float:
        """Compute a graph‑based loss over a batch of outputs."""
        activations = feedforward(self.architecture, self.weights, samples)
        final_states = [acts[-1] for acts in activations]
        G = fidelity_adjacency(final_states, threshold)
        if G.number_of_edges() == 0:
            return 0.0
        avg_weight = sum(data["weight"] for _, _, data in G.edges(data=True)) / G.number_of_edges()
        return 1.0 - avg_weight

    def __repr__(self) -> str:
        return f"<GraphQNN arch={self.architecture} weights={len(self.weights)}>"


__all__ = [
    "GraphQNN",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "random_training_data",
]
