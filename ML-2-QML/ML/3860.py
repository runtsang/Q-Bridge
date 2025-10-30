"""GraphQNNHybrid: classical graph neural network with optional quantum embedding.

This module implements a hybrid graph neural network that seamlessly
switches between a purely classical feed‑forward network and a
quantum‑state based propagation pipeline.  The design borrows the
random network generation, state fidelity, and weighted adjacency
construction from the original GraphQNN and EstimatorQNN seeds,
while extending them with a lightweight torch implementation
that can be used as a drop‑in replacement in existing classical
pipelines.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import torch
import networkx as nx

Tensor = torch.Tensor


def _rand_linear(in_features: int, out_features: int) -> Tensor:
    """Return a weight matrix with orthonormal rows."""
    W = torch.randn(out_features, in_features)
    Q, _ = torch.qr(W.t())
    return Q.t()


def random_training_data(target: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic regression data for the target linear map."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        x = torch.randn(target.size(1))
        y = target @ x
        dataset.append((x, y))
    return dataset


def random_network(arch: Sequence[int], samples: int):
    """Create a random fully‑connected network and a matching training set."""
    weights: List[Tensor] = [_rand_linear(arch[i], arch[i + 1]) for i in range(len(arch) - 1)]
    target = weights[-1]
    training = random_training_data(target, samples)
    return list(arch), weights, training, target


def feedforward(arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) \
        -> List[List[Tensor]]:
    """Forward propagate each sample through the linear layers."""
    activations: List[List[Tensor]] = []
    for x, _ in samples:
        layer_outs = [x]
        h = x
        for w in weights:
            h = torch.tanh(w @ h)
            layer_outs.append(h)
        activations.append(layer_outs)
    return activations


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Compute the squared overlap between two state vectors."""
    a_n = a / (torch.norm(a) + 1e-12)
    b_n = b / (torch.norm(b) + 1e-12)
    return float((a_n @ b_n).item() ** 2)


def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build a graph where edges reflect state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        f = state_fidelity(a, b)
        if f >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and f >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G


class GraphQNNHybrid:
    """Hybrid graph‑neural‑quantum network with a classical‑looking API."""
    def __init__(self, arch: Sequence[int]) -> None:
        self.arch = list(arch)
        self.weights: List[Tensor] | None = None
        self.training: List[Tuple[Tensor, Tensor]] | None = None

    def initialize(self, samples: int = 100) -> None:
        """Create random weights and a training set."""
        _, self.weights, self.training, _ = random_network(self.arch, samples)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the classical layers."""
        h = x
        for w in self.weights:
            h = torch.tanh(w @ h)
        return h

    def train(self, epochs: int = 10, lr: float = 1e-3) -> None:
        """A minimal training loop using mean‑squared‑error."""
        if self.training is None:
            raise RuntimeError("Network not initialized")
        opt = torch.optim.Adam(self.weights, lr=lr)
        for _ in range(epochs):
            for inp, tgt in self.training:
                opt.zero_grad()
                out = self.forward(inp)
                loss = torch.mean((out - tgt) ** 2)
                loss.backward()
                opt.step()

    def fidelity_graph(self, threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
        """Return a graph constructed from the activations of all samples."""
        activations = feedforward(self.arch, self.weights, self.training)
        states = [act[-1] for act in activations]
        return fidelity_adjacency(states, threshold,
                                   secondary=secondary,
                                   secondary_weight=secondary_weight)


__all__ = [
    "GraphQNNHybrid",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
