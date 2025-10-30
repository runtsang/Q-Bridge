"""Hybrid Graph Neural Network – Classical side.

Implements a lightweight training loop that mirrors the original API but adds
gradient‑based optimisation.  The public interface is deliberately
identical to the seed so that existing experiments continue to run unchanged.

"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence

import networkx as nx
import torch
from torch import nn, optim

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a learnable weight matrix of shape (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32, requires_grad=True)


def random_training_data(weight: Tensor, samples: int) -> List[tuple[Tensor, Tensor]]:
    """Generate synthetic (feature, target) pairs for a linear target."""
    dataset: List[tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1))
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create architecture, learnable weights, training data and the target weight."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Return activations for each sample in a list of lists."""
    activations_all: List[List[Tensor]] = []
    for features, _ in samples:
        activations: List[Tensor] = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        activations_all.append(activations)
    return activations_all


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Normalized squared dot product between two vectors."""
    na = a / (torch.norm(a) + 1e-12)
    nb = b / (torch.norm(b) + 1e-12)
    return float((na @ nb).item() ** 2)


def fidelity_adjacency(
    states: List[Tensor],
    threshold: float,
    *,
    secondary: None | float = None,
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


class HybridGraphQNN:
    """Convenience wrapper that stores architecture, weights and an SGD optimiser.

    The class exposes a minimal API:
        * ``train_step`` – performs one gradient‑descent step on a batch.
        * ``fidelity_graph`` – builds a weighted graph from hidden states.
        * ``predict`` – returns the final layer activations for a single input.
    """

    def __init__(self, arch: Sequence[int], lr: float = 1e-3):
        self.arch = list(arch)
        self.weights: List[Tensor] = [
            _random_linear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:])
        ]
        self.optimizer = optim.Adam(self.weights, lr=lr)

    def forward(self, x: Tensor) -> List[Tensor]:
        """Return the list of activations for a single input."""
        activations: List[Tensor] = [x]
        current = x
        for w in self.weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        return activations

    def train_step(self, batch: Iterable[tuple[Tensor, Tensor]]) -> float:
        """Perform one SGD step on a batch and return the average MSE loss."""
        self.optimizer.zero_grad()
        loss = 0.0
        for inp, target in batch:
            activations = self.forward(inp)
            pred = activations[-1]
            loss += nn.functional.mse_loss(pred, target, reduction="sum")
        loss /= len(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fidelity_graph(self, batch: Iterable[tuple[Tensor, Tensor]], threshold: float) -> nx.Graph:
        """Return a graph built from the hidden states of a batch."""
        hidden_states = [self.forward(inp)[1] for inp, _ in batch]
        return fidelity_adjacency(hidden_states, threshold)


__all__ = [
    "HybridGraphQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
