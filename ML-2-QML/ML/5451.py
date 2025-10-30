"""Combined classical graph neural network utilities with ancillary modules.

This module merges the classical GNN utilities from the seed
GraphQNN.py, the hybrid Quanvolution components, the classifier
factory, and the sampler network.  The resulting :class:`GraphQNNGen345`
exposes a unified interface that can be used in pure‑classical
experiments while still offering the same API as its quantum
counterpart.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
#  Classical GNN utilities
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Generate a random weight matrix of shape (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Create synthetic pairs (x, Wx) for supervised learning."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Return architecture, weight list, training data, and target weight."""
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
    """Propagate inputs through the classical network and return layerwise states."""
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
    """Squared overlap of two normalized vectors."""
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
#  Quanvolution components (classical)
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    """Simple 2×2 kernel convolution that reshapes output to a flat vector."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """Classical classifier built on the QuanvolutionFilter."""

    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


# --------------------------------------------------------------------------- #
#  Classical classifier factory
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward classifier and return metadata.
    Mirrors the signature of the quantum counterpart.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


# --------------------------------------------------------------------------- #
#  Classical sampler network
# --------------------------------------------------------------------------- #
def SamplerQNN() -> nn.Module:
    """Simple feed‑forward sampler returning a probability distribution."""
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return F.softmax(self.net(inputs), dim=-1)

    return SamplerModule()


# --------------------------------------------------------------------------- #
#  Unified GraphQNNGen345 class
# --------------------------------------------------------------------------- #
class GraphQNNGen345:
    """
    A hybrid interface that bundles classical GNN utilities, quanvolution,
    classifier construction, and sampler logic.  The class is intentionally
    stateless; all methods are classmethods or staticmethods to mirror the
    original seed design while providing a single point of import.
    """

    # GNN utilities
    random_network = staticmethod(random_network)
    feedforward = staticmethod(feedforward)
    fidelity_adjacency = staticmethod(fidelity_adjacency)
    state_fidelity = staticmethod(state_fidelity)

    # Quanvolution components
    QuanvolutionFilter = QuanvolutionFilter
    QuanvolutionClassifier = QuanvolutionClassifier

    # Classifier factory
    build_classifier_circuit = staticmethod(build_classifier_circuit)

    # Sampler
    SamplerQNN = staticmethod(SamplerQNN)

    __all__ = [
        "random_network",
        "feedforward",
        "fidelity_adjacency",
        "state_fidelity",
        "QuanvolutionFilter",
        "QuanvolutionClassifier",
        "build_classifier_circuit",
        "SamplerQNN",
    ]
