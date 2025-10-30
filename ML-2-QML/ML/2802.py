"""Hybrid classical graph neural network with optional convolutional encoder.

The module reproduces the GraphQNN utilities while adding a lightweight CNN encoder
to mimic the Quantum‑NAT feature extractor.  All functions are compatible with the
original GraphQNN API, enabling seamless replacement or extension in downstream pipelines.

The class ``GraphQNNHybrid`` is a PyTorch ``nn.Module`` that accepts an architecture
sequence (e.g. ``[16, 32, 64]``) and builds a linear stack with tanh activations.
A two‑layer CNN is placed before the linear stack to produce a 4‑dimensional embedding,
mirroring the Quantum‑NAT encoder.  The module also provides static helpers for
generating random networks, training data, and fidelity‑based adjacency graphs.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
#  Core utilities (identical to the original GraphQNN module)
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix for a dense layer."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate input–output pairs for a linear target function."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Construct a random classical network and training data."""
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
    """Run a forward pass through a classical feed‑forward network."""
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
    """Squared overlap between two classical weight vectors."""
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
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#  Hybrid classical model
# --------------------------------------------------------------------------- #
class GraphQNNHybrid(nn.Module):
    """A hybrid graph‑based neural network with a convolutional encoder.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes for the linear stack (e.g. ``[16, 32, 64]``).
    conv_features : int, optional
        Number of output channels for the first convolutional layer.
    conv_kernel : int, optional
        Kernel size for the convolutional layers.
    """

    def __init__(
        self,
        arch: Sequence[int],
        conv_features: int = 8,
        conv_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.arch = list(arch)

        # Convolutional encoder (Quantum‑NAT style)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, conv_features, kernel_size=conv_kernel, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_features, conv_features * 2, kernel_size=conv_kernel, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Linear stack
        layers: List[nn.Module] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.Tanh())
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through encoder + linear stack."""
        bsz = x.shape[0]
        x = self.encoder(x)
        x = x.view(bsz, -1)
        return self.classifier(x)

    # --------------------------------------------------------------------- #
    #  Compatibility helpers – identical to the original GraphQNN utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def random_network(*args, **kwargs):
        return random_network(*args, **kwargs)

    @staticmethod
    def random_training_data(*args, **kwargs):
        return random_training_data(*args, **kwargs)

    @staticmethod
    def feedforward(*args, **kwargs):
        return feedforward(*args, **kwargs)

    @staticmethod
    def state_fidelity(*args, **kwargs):
        return state_fidelity(*args, **kwargs)

    @staticmethod
    def fidelity_adjacency(*args, **kwargs):
        return fidelity_adjacency(*args, **kwargs)


__all__ = [
    "GraphQNNHybrid",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
