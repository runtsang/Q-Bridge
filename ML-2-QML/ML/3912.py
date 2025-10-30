"""Hybrid Graph Neural Network – Classical implementation.

The public API mirrors the original GraphQNN module but adds a
convolutional filter that can be used as a drop‑in replacement for the
quantum quanvolution layer.  All functions are pure and stateless
except for the :class:`GraphQNNGen224` instance which stores the
architecture and optional weight tensors.

Typical usage:

    >>> from GraphQNN__gen224 import GraphQNNGen224
    >>> gnn = GraphQNNGen224([4, 8, 4])
    >>> arch, weights, train, target = gnn.random_network(samples=10)
    >>> activations = gnn.feedforward(arch, weights, train)
    >>> gnn.fidelity_adjacency([a[-1] for a in activations], 0.9)

"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix with shape (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def _random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic supervised data for a linear target."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


class GraphQNNGen224:
    """Container for a classical graph neural network architecture.

    Parameters
    ----------
    qnn_arch
        Sequence of layer widths, e.g. ``[4, 8, 4]``.
    """

    def __init__(self, qnn_arch: Sequence[int]) -> None:
        self.qnn_arch = list(qnn_arch)

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Return architecture, random weights, training data and target weight.

        The target weight is the last layer; it is used by the training data
        generator so that synthetic examples match a linear model.
        """
        weights: List[Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(_random_linear(in_f, out_f))
        target_weight = weights[-1]
        training_data = _random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Propagate all samples through the network.

        Returns a list of activation lists (including the input) for each sample.
        """
        activations: List[List[Tensor]] = []
        for features, _ in samples:
            layer_outputs = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                layer_outputs.append(current)
            activations.append(layer_outputs)
        return activations

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Compute squared overlap between two vectors, normalised.

        This mirrors the quantum fidelity used in the QML implementation.
        """
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from pairwise state fidelities.

        Edges with fidelity ≥ ``threshold`` receive weight 1; if a
        ``secondary`` bound is provided, fidelities between ``secondary`` and
        ``threshold`` receive ``secondary_weight``.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen224.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def random_training_data(weight: Tensor, samples: int):
        """Alias for compatibility with the QML version."""
        return _random_training_data(weight, samples)

    @staticmethod
    def conv_filter(kernel_size: int = 2, threshold: float = 0.0):
        """Return a classical convolution filter.

        The filter is a thin wrapper around :class:`torch.nn.Conv2d` that
        mimics the interface of the quantum quanvolution circuit.
        """
        from.Conv import Conv as ClassicalConv
        return ClassicalConv()

__all__ = [
    "GraphQNNGen224",
]
