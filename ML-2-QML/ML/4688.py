"""Hybrid classical‑quantum utilities for classification and graph‑based analysis.

This file contains three self‑contained components that mirror the API of the
reference seed modules while combining their strengths:

* :func:`build_classifier_circuit` – constructs a classical feed‑forward
  network that outputs logits for two classes.
* :class:`SamplerQNN` – a lightweight torch module that mimics the
  quantum‑sampler interface but operates purely classically.
* :class:`GraphQNNUtility` – a helper that builds a graph from fidelity
  between state vectors produced by the network, providing utilities for
  random network generation and feed‑forward evaluation.

All components expose a ``forward`` or equivalent method that is
compatible with downstream code expecting the seed modules.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Sequence
import itertools
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --------------------------------------------------------------------------- #
# 1. Classical classifier circuit
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Build a classical feed‑forward network that mirrors the quantum helper
    interface.

    Parameters
    ----------
    num_features : int
        Number of input features (also the hidden layer dimension).
    depth : int
        Number of hidden layers.

    Returns
    -------
    network : nn.Module
        Sequential network with hidden ReLU layers and a 2‑output head.
    encoding : Iterable[int]
        Indices of the feature dimensions used for encoding; identical to
        ``range(num_features)``.
    weight_sizes : Iterable[int]
        Number of trainable parameters for each linear layer.
    observables : List[int]
        Dummy observables (just indices) to be compatible with the quantum
        version.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
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
    encoding = list(range(num_features))
    observables = [0, 1]  # placeholder indices for 2 classes
    return network, encoding, weight_sizes, observables


# --------------------------------------------------------------------------- #
# 2. Classical sampler network
# --------------------------------------------------------------------------- #
class SamplerQNN(nn.Module):
    """A minimal torch implementation of a sampler network."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return a softmax‑normalised probability distribution."""
        return F.softmax(self.net(inputs), dim=-1)


def SamplerQNN_factory() -> SamplerQNN:
    """Factory that mirrors the original seed signature."""
    return SamplerQNN()


# --------------------------------------------------------------------------- #
# 3. Graph‑based quantum neural network utilities (classical stub)
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate synthetic training data for the given target weight."""
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Construct a random weight matrix chain and synthetic data."""
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
    """Run a forward pass through the network and record all layer activations."""
    activations: List[List[torch.Tensor]] = []
    for features, _ in samples:
        layer_acts = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            layer_acts.append(current)
        activations.append(layer_acts)
    return activations


def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap between two state vectors."""
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


__all__ = [
    "build_classifier_circuit",
    "SamplerQNN",
    "SamplerQNN_factory",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
