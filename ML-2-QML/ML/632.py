"""Hybrid classical Graph Neural Network utilities.

This module extends the original GraphQNN interface by providing a
`GraphQNN` class that stores architecture and weight matrices,
and exposes class‑method helpers for random network generation and
training data creation.  The implementation is fully classical
(NumPy/Torch) and retains the original feed‑forward and fidelity
functions for backward compatibility.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional

import networkx as nx
import numpy as np
import torch

Tensor = torch.Tensor


class GraphQNN:
    """Classical graph neural network with feed‑forward and fidelity utilities.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes of the network.
    weights : Optional[List[Tensor]]
        List of weight matrices.  If ``None`` random matrices are generated.
    """

    def __init__(self, arch: Sequence[int], weights: Optional[List[Tensor]] = None):
        self.arch = list(arch)
        self.weights = (
            weights
            or [
                self._random_linear(in_f, out_f)
                for in_f, out_f in zip(arch[:-1], arch[1:])
            ]
        )

    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        """Return a random weight matrix."""
        return torch.randn(out_features, in_features, dtype=torch.float32)

    def feedforward(
        self, samples: Iterable[Tuple[Tensor, Tensor]]
    ) -> List[List[Tensor]]:
        """Propagate every sample through the network.

        Parameters
        ----------
        samples : Iterable[Tuple[Tensor, Tensor]]
            Iterable yielding ``(features, target)`` pairs.  The target is
            ignored by the forward pass but kept for API compatibility.

        Returns
        -------
        List[List[Tensor]]
            List of activation lists per sample.  Each inner list contains
            the input state followed by the activations of every layer.
        """
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in self.weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def random_training_data(
        weight: Tensor, samples: int
    ) -> List[Tuple[Tensor, Tensor]]:
        """Generate synthetic training data for a linear map."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(
        arch: Sequence[int], samples: int
    ) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Generate a random network together with training data.

        Returns
        -------
        arch : List[int]
            The architecture.
        weights : List[Tensor]
            Weight matrices.
        training_data : List[Tuple[Tensor, Tensor]]
            Synthetic dataset.
        target_weight : Tensor
            The last weight matrix (used as ground truth).
        """
        weights: List[Tensor] = [
            GraphQNN._random_linear(in_f, out_f)
            for in_f, out_f in zip(arch[:-1], arch[1:])
        ]
        target_weight = weights[-1]
        training_data = GraphQNN.random_training_data(target_weight, samples)
        return list(arch), weights, training_data, target_weight

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Squared overlap between two state vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Create an adjacency graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


# Backward‑compatible wrappers ------------------------------------------------

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Wrapper that forwards to :class:`GraphQNN`."""
    return GraphQNN(qnn_arch, list(weights)).feedforward(samples)


def random_network(
    qnn_arch: Sequence[int], samples: int
) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Wrapper for :meth:`GraphQNN.random_network`."""
    return GraphQNN.random_network(qnn_arch, samples)


def random_training_data(
    weight: Tensor, samples: int
) -> List[Tuple[Tensor, Tensor]]:
    """Wrapper for :meth:`GraphQNN.random_training_data`."""
    return GraphQNN.random_training_data(weight, samples)


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Wrapper for :meth:`GraphQNN.state_fidelity`."""
    return GraphQNN.state_fidelity(a, b)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: Optional[float] = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Wrapper for :meth:`GraphQNN.fidelity_adjacency`."""
    return GraphQNN.fidelity_adjacency(
        states, threshold, secondary=secondary, secondary_weight=secondary_weight
    )
