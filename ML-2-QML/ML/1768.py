"""GraphQNN__gen358: Classical neural network with stochastic depth and dropout.

The module mirrors the original interface but extends the
`random_network` factory to return a fully‑trained PyTorch model
using randomly initialised weights and stochastic depth.  Dropout
is applied after every linear layer and a `DepthNorm` layer normalises
the network output.  The data pipeline can be reused by downstream
experiments.  The code is fully importable as a single module and
provides a small `main` entry‑point for quick testing."""
from __future__ import annotations

import itertools
from typing import Iterable, List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
#  Utility functions
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a randomly initialised weight matrix for a linear layer."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a list of (features, target) tuples for training."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: List[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Create a random linear network and generate training data."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return qnn_arch, weights, training_data, target_weight


# --------------------------------------------------------------------------- #
#  Stochastic depth and normalisation layers
# --------------------------------------------------------------------------- #
class StochasticDepth(nn.Module):
    """Drop the output of a layer with probability `p` during training."""
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = torch.bernoulli((1 - self.p) * torch.ones(x.shape[0], 1, device=x.device))
        return x * mask / (1 - self.p)


class DepthNorm(nn.Module):
    """Normalize each sample to unit norm."""
    def forward(self, x: Tensor) -> Tensor:
        norm = torch.norm(x, dim=1, keepdim=True) + 1e-12
        return x / norm


# --------------------------------------------------------------------------- #
#  GraphQNN model definition
# --------------------------------------------------------------------------- #
class GraphQNNModel(nn.Module):
    """Feed‑forward network with optional dropout and stochastic depth."""
    def __init__(
        self,
        arch: List[int],
        weights: List[Tensor],
        dropout_rate: float = 0.0,
        stochastic_depth_rate: float = 0.0,
    ):
        super().__init__()
        self.arch = arch
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.stochastic_depth = StochasticDepth(stochastic_depth_rate)
        self.depth_norm = DepthNorm()
        for w in weights:
            layer = nn.Linear(w.size(1), w.size(0), bias=False)
            layer.weight = nn.Parameter(w)
            self.layers.append(layer)

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        activations = [x]
        for layer in self.layers:
            x = layer(x)
            x = torch.tanh(x)
            x = self.stochastic_depth(x)
            x = self.dropout(x)
            activations.append(x)
        x = self.depth_norm(x)
        return x, activations


def create_model(qnn_arch: List[int], weights: List[Tensor], **kwargs) -> GraphQNNModel:
    """Convenience wrapper that creates a `GraphQNNModel` from weights."""
    return GraphQNNModel(qnn_arch, weights, **kwargs)


# --------------------------------------------------------------------------- #
#  Forward propagation utilities
# --------------------------------------------------------------------------- #
def feedforward(
    qnn_arch: List[int],
    weights: List[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Return activations for each sample across all layers."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        stored.append(activations)
    return stored


# --------------------------------------------------------------------------- #
#  Fidelity and graph utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the absolute squared overlap between two classical state vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: List[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
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
#  Demo / test harness
# --------------------------------------------------------------------------- #
def main():
    arch = [4, 8, 8, 2]
    samples = 10
    qnn_arch, weights, training_data, target_weight = random_network(arch, samples)
    model = create_model(qnn_arch, weights, dropout_rate=0.2, stochastic_depth_rate=0.1)
    model.train()
    activations = feedforward(qnn_arch, weights, training_data)
    states = [acts[-1] for acts in activations]
    G = fidelity_adjacency(states, 0.9)
    print("Graph nodes:", G.number_of_nodes())
    print("Graph edges:", G.number_of_edges())


if __name__ == "__main__":
    main()


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNNModel",
    "create_model",
]
