"""Hybrid Graph Neural Network combining classical feedforward and optional convolution filter.

This module unifies the classical GraphQNN utilities with a lightweight
convolutional front‑end inspired by the Conv.py seed.  It exposes a
`GraphQNNHybrid` class that can be instantiated, built with random weights,
and run on input tensors.  The implementation follows the same public API
as the quantum counterpart, enabling side‑by‑side experiments.

Key extensions over the seed:
- Optional Conv filter that can be applied before the linear layers.
- Unified adjacency graph construction based on feature similarity.
- Public attributes for the architecture and the trained weights.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
from torch import nn

# Import the classical Conv filter from Conv.py seed
try:
    from Conv import Conv  # type: ignore
except Exception:  # pragma: no cover
    # Minimal fallback if Conv cannot be imported
    def Conv():
        class DummyConv:
            def run(self, data):
                return 0.0
        return DummyConv()


Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic regression data targeting the given weight matrix."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a chain of random linear layers and a training set."""
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
    """Run a forward pass through the linear chain, storing activations."""
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
    """Squared overlap of two feature vectors."""
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
    """Build a graph from pairwise feature similarities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class GraphQNNHybrid:
    """Hybrid classical graph neural network with optional convolution front‑end."""

    def __init__(
        self,
        arch: Sequence[int],
        conv_kernel: int = 2,
        conv_threshold: float = 0.0,
    ) -> None:
        self.arch = list(arch)
        self.conv = Conv()  # returns a ConvFilter instance
        # Adjust kernel/threshold if possible
        if hasattr(self.conv, "kernel_size"):
            self.conv.kernel_size = conv_kernel
        if hasattr(self.conv, "threshold"):
            self.conv.threshold = conv_threshold
        if hasattr(self.conv, "conv"):
            self.conv.conv = nn.Conv2d(1, 1, kernel_size=conv_kernel, bias=True)
        self.weights: List[Tensor] | None = None
        self.training_data: List[Tuple[Tensor, Tensor]] | None = None
        self.target_weight: Tensor | None = None

    def build_random(self, samples: int) -> None:
        """Generate random weights and synthetic training data."""
        _, self.weights, self.training_data, self.target_weight = random_network(
            self.arch, samples
        )

    def run(self, data: Tensor) -> List[Tensor]:
        """Forward pass through conv filter (if enabled) followed by the linear chain."""
        if self.weights is None:
            raise RuntimeError("Model has not been built; call `build_random` first.")
        # Apply the convolutional filter to the input data if it is 2‑D.
        conv_out = self.conv.run(data.detach().cpu().numpy())
        # Concatenate the scalar conv output to the original feature vector.
        features = torch.cat([data, torch.tensor([conv_out], dtype=torch.float32)])
        # Forward through the linear layers.
        activations = [features]
        current = features
        for weight in self.weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        return activations

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Convenience wrapper around the shared utility."""
        return fidelity_adjacency(
            states, threshold, secondary=secondary, secondary_weight=secondary_weight
        )

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Convenience wrapper around the shared utility."""
        return state_fidelity(a, b)


__all__ = [
    "GraphQNNHybrid",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
