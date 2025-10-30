"""Hybrid classical graph neural network utilities with quantum‑inspired extensions.

This module builds on the original GraphQNN.py by adding support for hybrid
classical‑quantum layers, fully‑connected and convolution‑pooling patterns,
and a unified adjacency construction.  The public API mirrors the seed
functions (`feedforward`, `random_network`, `fidelity_adjacency`, etc.) but
now accepts both classical tensors and quantum states through a shared
`GraphQNNGen180` class.

Key additions:
* `FCL()` – a lightweight fully‑connected PyTorch layer that mimics the
  quantum example in the reference pair.
* `QCNN()` – a classical QCNN‑style network (placeholder) that can be
  used as a drop‑in replacement for the quantum version.
* `GraphQNNGen180` – a class that holds architecture, weights, and
  training data and exposes `feedforward`, `hybrid_forward`,
  `get_adjacency`, and a static `random_network` helper.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple, List, Union

import networkx as nx
import torch
import numpy as np

Tensor = torch.Tensor
State = Union[Tensor, np.ndarray]  # for compatibility with QCNN's numpy output


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training pairs (x, Wx) for a linear layer."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Generate a random classical network and its training data."""
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
    """Compute the activations of a purely classical feed‑forward network."""
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
    """Cosine similarity of two classical feature vectors."""
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


class GraphQNNGen180:
    """
    Hybrid classical graph neural network.

    Parameters
    ----------
    arch : Sequence[int]
        Node counts per layer.  Even indices are treated as classical
        layers, odd indices as quantum layers.  The class will generate
        random weights/unities accordingly.
    """

    def __init__(self, arch: Sequence[int]):
        self.arch = list(arch)
        self.classical_weights: List[Tensor] = []
        self.quantum_units: List[List[Tensor]] = []  # placeholder for future quantum support
        self.training_data: List[Tuple[Tensor, Tensor]] = []

        # Build classical weights for even layers
        for i in range(0, len(arch) - 1, 2):
            self.classical_weights.append(_random_linear(arch[i], arch[i + 1]))

        # Build training data using the last classical weight
        if self.classical_weights:
            target = self.classical_weights[-1]
            self.training_data = random_training_data(target, 100)

    def feedforward(self, inputs: Tensor) -> List[Tensor]:
        """Run a classical feed‑forward pass."""
        activations = [inputs]
        current = inputs
        for weight in self.classical_weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        return activations

    def hybrid_forward(self, inputs: Tensor) -> List[Tensor]:
        """Alternate classical and quantum layers (quantum part is stubbed)."""
        activations = [inputs]
        current = inputs
        for idx, weight in enumerate(self.classical_weights):
            current = torch.tanh(weight @ current)
            activations.append(current)
            # quantum placeholder: no operation
            # In a full implementation we would apply a unitary here.
        return activations

    def get_adjacency(self, threshold: float, *, secondary: float | None = None) -> nx.Graph:
        """Construct a graph from the activations of the training data."""
        states = [self.feedforward(state)[-1] for state, _ in self.training_data]
        return fidelity_adjacency(states, threshold, secondary=secondary)

    @staticmethod
    def random_network(arch: Sequence[int], samples: int):
        """Convenience wrapper that returns a fully initialized GraphQNNGen180."""
        return GraphQNNGen180(arch)


def FCL(n_features: int = 1):
    """Return a simple fully‑connected PyTorch layer."""
    class FullyConnectedLayer(torch.nn.Module):
        def __init__(self, n_features: int = 1) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(n_features, 1)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().numpy()

    return FullyConnectedLayer()


def QCNN():
    """Return a classical QCNN‑style network (placeholder)."""
    return GraphQNNGen180([8, 16, 16, 12, 8, 4, 4, 1])


__all__ = [
    "GraphQNNGen180",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "FCL",
    "QCNN",
]
