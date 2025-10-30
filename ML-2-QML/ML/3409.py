"""Graph-based neural network utilities with a PyTorch backend.

Provides:
- Random graph‑structured neural network generation.
- Classical feed‑forward propagation with tanh activations.
- Fidelity‑based graph construction for state similarity.
- A lightweight EstimatorQNN wrapper for regression.

The class `GraphQNNGen019` encapsulates all functionality and mirrors
the public API of the original GraphQNN module while extending it with
quantum‑aware utilities for experimentation.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix with shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic `(x, Wx)` pairs for the target weight."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random graph‑structured network and a training set."""
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
    """Return activations for each sample through the network."""
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
    """Squared overlap between two state vectors."""
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
    """Build a weighted graph from pairwise state fidelities."""
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
# EstimatorQNN – lightweight PyTorch regressor
# --------------------------------------------------------------------------- #
def EstimatorQNN() -> torch.nn.Module:
    """Return a simple fully‑connected regression network."""
    class EstimatorNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(2, 8),
                torch.nn.Tanh(),
                torch.nn.Linear(8, 4),
                torch.nn.Tanh(),
                torch.nn.Linear(4, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(inputs)

    return EstimatorNN()


# --------------------------------------------------------------------------- #
# Unified class interface
# --------------------------------------------------------------------------- #
class GraphQNNGen019:
    """Convenience wrapper exposing classical and estimator utilities."""

    def __init__(self, arch: Sequence[int] | None = None) -> None:
        if arch is None:
            arch = [2, 4, 4, 1]
        self.arch = list(arch)

    def random_network(self, samples: int = 100):
        return random_network(self.arch, samples)

    def feedforward(self, weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]):
        return feedforward(self.arch, weights, samples)

    def fidelity_adjacency(self, states: Sequence[Tensor], threshold: float, **kwargs):
        return fidelity_adjacency(states, threshold, **kwargs)

    def estimator(self) -> torch.nn.Module:
        return EstimatorQNN()


__all__ = [
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "EstimatorQNN",
    "GraphQNNGen019",
]
