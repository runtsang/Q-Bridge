"""QuantumHybridNet – classical implementation.

This module defines a hybrid neural network that merges ideas from
four seed projects: a CNN + fully‑connected head, a shallow
classifier factory, a fully‑connected layer stand‑in, and
graph‑based utilities.  The public API is a single class
``QuantumHybridNet`` that can optionally attach a classifier or
build a fidelity graph from the latent representations.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# GraphQNN utilities – classical analogues
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    """Generate a random weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Produce synthetic data for a linear target."""
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Build a random layered network with linear weights."""
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """Apply the network to a collection of inputs."""
    stored: List[List[torch.Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap of two unit‑norm vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from pairwise fidelities."""
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
# Classifier factory – classical
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Return a shallow fully‑connected network and metadata."""
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
# Fully‑connected layer stand‑in
# --------------------------------------------------------------------------- #
def FCL() -> nn.Module:
    """A tiny fully‑connected layer that mimics the quantum FCL stand‑in."""
    class FullyConnectedLayer(nn.Module):
        def __init__(self, n_features: int = 1) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: Iterable[float]) -> torch.Tensor:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach()

    return FullyConnectedLayer()

# --------------------------------------------------------------------------- #
# QuantumHybridNet – classical
# --------------------------------------------------------------------------- #
class QuantumHybridNet(nn.Module):
    """Hybrid classical network that combines CNN feature extraction,
    a fully‑connected head, and an optional classifier.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 1).
    out_features : int
        Size of the latent vector fed to the classifier (default 4).
    use_classifier : bool
        If ``True`` a shallow classifier built by :func:`build_classifier_circuit`
        is appended after the fully‑connected head.
    use_graph : bool
        When ``True`` the latent vectors are turned into a fidelity‑based
        graph; the graph is returned together with the logits so that
        downstream algorithms can exploit relational structure.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 4,
        use_classifier: bool = True,
        use_graph: bool = False,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, out_features)
        )
        self.norm = nn.BatchNorm1d(out_features)

        self.use_classifier = use_classifier
        self.use_graph = use_graph

        if self.use_classifier:
            self.classifier, _, _, _ = build_classifier_circuit(
                num_features=out_features, depth=2
            )
        if self.use_graph:
            self.graph_threshold = 0.8

    def forward(self, x: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, nx.Graph]:
        """Forward pass returning logits and optionally an adjacency graph."""
        bsz = x.shape[0]
        feats = self.features(x)
        flattened = feats.view(bsz, -1)
        latent = self.fc(flattened)
        logits = self.norm(latent)

        if self.use_classifier:
            logits = self.classifier(logits)

        if self.use_graph:
            # Build a simple fidelity graph over the batch of logits.
            adj = fidelity_adjacency(
                [logit.squeeze() for logit in logits], self.graph_threshold
            )
            return logits, adj

        return logits

__all__ = [
    "QuantumHybridNet",
    "build_classifier_circuit",
    "FCL",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
