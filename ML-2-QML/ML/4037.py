"""
Unified hybrid classifier with a pure‑classical backbone.
The implementation deliberately mirrors the quantum helper interface so that
`build_classifier_circuit` can be called from either side.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools

# --------------------------------------------------------------------------- #
#  Classical building blocks
# --------------------------------------------------------------------------- #
def build_classifier_circuit(
    num_features: int,
    depth: int,
    dropout: float = 0.0,
    batchnorm: bool = False,
) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """
    Construct a feed‑forward classifier with optional dropout and batch‑norm.
    Returns a tuple containing the network, the encoding indices, the
    weight‑sizes of each linear layer, and the output observables (class labels).
    """
    layers: List[nn.Module] = []

    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features, bias=True)
        layers.append(linear)
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())

        if batchnorm:
            layers.append(nn.BatchNorm1d(num_features))
        layers.append(nn.ReLU(inplace=True))

        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = [0, 1]  # class indices
    return network, encoding, weight_sizes, observables


# --------------------------------------------------------------------------- #
#  Graph‑based utilities
# --------------------------------------------------------------------------- #
def random_training_data(
    weight: torch.Tensor, samples: int
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate a synthetic regression dataset from a linear weight."""
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(
    qnn_arch: List[int], samples: int
) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """
    Build a random weight matrix sequence and a synthetic training set.
    Returns architecture, weight list, training data, and the last layer weight.
    """
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return qnn_arch, weights, training_data, target_weight


def feedforward(
    qnn_arch: List[int],
    weights: List[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """
    Forward‑propagate a batch of samples through the linear network.
    Returns a list of activation lists for each sample.
    """
    activations: List[List[torch.Tensor]] = []
    for features, _ in samples:
        layerwise: List[torch.Tensor] = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            layerwise.append(current)
        activations.append(layerwise)
    return activations


def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the squared overlap of two unit‑norm vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: List[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """
    Build a weighted graph where edges exist if the fidelity between two
    states exceeds *threshold* (weight 1) or *secondary* (weight *secondary_weight*).
    """
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
#  Hybrid wrapper
# --------------------------------------------------------------------------- #
class UnifiedQuantumClassifier(nn.Module):
    """
    Hybrid network that optionally prefixes a quantum feature extractor.
    If *quantum_fn* is provided it must be a callable that accepts a
    torch.Tensor of shape (batch, features) and returns a torch.Tensor
    of quantum‑derived features. The returned tensor is fed into the
    classical classifier defined by *build_classifier_circuit*.
    """
    def __init__(
        self,
        num_features: int,
        depth: int,
        dropout: float = 0.0,
        batchnorm: bool = False,
        quantum_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        super().__init__()
        self.quantum_fn = quantum_fn
        self.classifier, _, _, _ = build_classifier_circuit(
            num_features, depth, dropout, batchnorm
        )
        self._num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantum_fn is not None:
            x = self.quantum_fn(x)
        return self.classifier(x)


__all__ = [
    "build_classifier_circuit",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "UnifiedQuantumClassifier",
]
