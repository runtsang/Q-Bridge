"""Graph-based quantum neural network with classical convolutional front‑end.

This module merges the graph‑based feed‑forward utilities of the original
GraphQNN with a classical convolutional filter that can be used as a
drop‑in replacement for a quantum quanvolution layer.  The class
GraphQNNGen220 exposes a unified API that can be used for both
classical and quantum experiments, but the ML implementation here
remains fully classical (NumPy / PyTorch).

Key features
------------
* Random network generation with fully‑connected linear layers.
* Classical convolutional filter (2×2) that can be applied before
  the graph network.
* Fidelity‑based adjacency graph construction.
* State‑fidelity computation for both feature vectors and quantum
  states (the latter is simply a placeholder returning 0.0).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
from torch import nn

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training pairs (x, Wx)."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random classical QNN and its training data."""
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
    """Run a forward pass through the linear graph network."""
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
    """Compute the squared cosine similarity between two feature vectors."""
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


class ConvFilter(nn.Module):
    """2×2 classical convolutional filter with sigmoid activation."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the filter and return the mean activation."""
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()


class GraphQNNGen220:
    """
    Unified classical graph neural network with an optional
    convolutional front‑end.  The class is intentionally lightweight
    and can be used as a drop‑in replacement for the original
    GraphQNN module while providing a clear separation between the
    classical and quantum halves.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        kernel_size: int = 2,
        use_conv: bool = True,
        conv_threshold: float = 0.0,
    ) -> None:
        self.qnn_arch = list(qnn_arch)
        self.kernel_size = kernel_size
        self.use_conv = use_conv
        self.conv = ConvFilter(kernel_size, conv_threshold) if use_conv else None

    def random_network(self, samples: int):
        """Generate a random network and training data."""
        return random_network(self.qnn_arch, samples)

    def random_training_data(self, weight: Tensor, samples: int):
        """Generate synthetic training data for the target layer."""
        return random_training_data(weight, samples)

    def feedforward(
        self,
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Run the full forward pass including an optional conv filter."""
        # If a conv filter is present, prepend its output as the first activation.
        processed_samples = []
        for features, target in samples:
            if self.use_conv:
                # Expect features to be a 2D array; convert to torch tensor.
                conv_out = self.conv(torch.as_tensor(features, dtype=torch.float32))
                # Concatenate conv output with original features.
                new_features = torch.cat([conv_out, features])
            else:
                new_features = features
            processed_samples.append((new_features, target))
        return feedforward(self.qnn_arch, weights, processed_samples)

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        return state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(
            states, threshold, secondary=secondary, secondary_weight=secondary_weight
        )
