"""Classical convolutional filter and graph neural network module.

This module defines ConvGraphQNN, a hybrid network that replaces the quantum
filter with a 2‑D convolution and then applies a graph‑based message passing
layer that uses cosine similarity to build the adjacency.  The implementation
is fully pure‑Python and uses PyTorch for tensor operations.
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
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
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
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class ConvGraphQNN(nn.Module):
    """Hybrid classical convolution + graph neural network.

    The network applies a 2‑D convolution to each kernel, then builds a graph
    of samples using cosine‑similarity based fidelity.  A single message‑
    passing step (average of neighbours) is used as a lightweight
    graph‑neuron.  The architecture is a drop‑in replacement for the
    original Conv/QML hybrid and is fully compatible with PyTorch.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        conv_threshold: float = 0.0,
        graph_threshold: float = 0.8,
        secondary_threshold: float | None = None,
        secondary_weight: float = 0.5,
        device: str = "cpu",
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=True,
        )
        self.conv_threshold = conv_threshold
        self.graph_threshold = graph_threshold
        self.secondary_threshold = secondary_threshold
        self.secondary_weight = secondary_weight
        self.device = device

    def forward(self, data: Sequence[torch.Tensor]) -> List[Tensor]:
        """Process a batch of 2‑D kernels.

        Args:
            data: Sequence of tensors each of shape (kernel_size, kernel_size).

        Returns:
            List[Tensor] of updated activations after graph propagation.
        """
        activations = []
        for kernel in data:
            tensor = kernel.view(1, 1, self.kernel_size, self.kernel_size).float()
            logits = self.conv(tensor)
            out = torch.sigmoid(logits - self.conv_threshold)
            activations.append(out.mean().squeeze())

        graph = fidelity_adjacency(
            activations,
            self.graph_threshold,
            secondary=self.secondary_threshold,
            secondary_weight=self.secondary_weight,
        )

        updated = []
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            if neighbors:
                neighbor_vals = torch.stack([activations[n] for n in neighbors])
                updated_val = torch.mean(neighbor_vals)
            else:
                updated_val = activations[node]
            updated.append(updated_val)

        return updated

__all__ = [
    "ConvGraphQNN",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
