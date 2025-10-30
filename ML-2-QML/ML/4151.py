"""Hybrid classical graph neural network that combines linear layers,
fidelity‑based adjacency, optional convolution feature extraction,
and a classical regression estimator.

The class `GraphQNNHybrid` mirrors the original GraphQNN API but
enhances it with a ConvFilter and a simple MLP estimator.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn

Tensor = torch.Tensor


# ---------- Utility functions ------------------------------------------------

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weights: Sequence[Tensor], samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training pairs using the last layer weights."""
    target = weights[-1]
    data: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(target.size(1), dtype=torch.float32)
        output = target @ features
        data.append((features, output))
    return data


def random_network(arch: Sequence[int], samples: int):
    """Create a random network and associated training data."""
    weights = [_random_linear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:])]
    training = random_training_data(weights, samples)
    return list(arch), weights, training, weights[-1]


def feedforward(arch: Sequence[int], weights: Sequence[Tensor], data: Iterable[Tuple[Tensor, Tensor]]):
    """Compute layer‑wise activations for each sample."""
    activations: List[List[Tensor]] = []
    for features, _ in data:
        layer_vals = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            layer_vals.append(current)
        activations.append(layer_vals)
    return activations


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two normalized vectors."""
    anorm = a / (torch.norm(a) + 1e-12)
    bnorm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(anorm, bnorm).item() ** 2)


def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build graph where edges correspond to state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G


# ---------- Convolution filter -----------------------------------------------

class ConvFilter(nn.Module):
    """Simple 2‑D convolution that mimics a quantum quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(x)
        act = torch.sigmoid(logits - self.threshold)
        return act.mean()


# ---------- Hybrid graph neural network class ---------------------------------

class GraphQNNHybrid(nn.Module):
    """
    Classical graph neural network that blends linear layers,
    fidelity‑based adjacency, and optional convolution feature extraction.
    """
    def __init__(self,
                 arch: Sequence[int],
                 conv_kernel: int = 2,
                 conv_threshold: float = 0.0,
                 use_fidelity: bool = True) -> None:
        super().__init__()
        self.arch = list(arch)
        self.use_fidelity = use_fidelity
        self.conv = ConvFilter(conv_kernel, conv_threshold)

        # Build linear layers
        layers: List[nn.Module] = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.Tanh())
        layers.pop()  # remove trailing activation
        self.net = nn.Sequential(*layers)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass over a single graph.
        node_features: shape (num_nodes, feature_dim)
        """
        # Convolution on each node's local patch
        conv_feats = torch.stack([self.conv(f) for f in node_features])
        # Linear propagation
        activations = [conv_feats]
        x = conv_feats
        for layer in self.net:
            x = layer(x)
            activations.append(x)
        return activations[-1]

    def build_adjacency(self, activations: torch.Tensor, threshold: float) -> nx.Graph:
        """Construct adjacency graph from node activations."""
        if not self.use_fidelity:
            return nx.Graph()
        states = [act.detach() for act in activations]
        return fidelity_adjacency(states, threshold)

    @staticmethod
    def random(seed_arch: Sequence[int], samples: int):
        """Convenience wrapper returning a random network and training data."""
        return random_network(seed_arch, samples)


__all__ = [
    "GraphQNNHybrid",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "ConvFilter",
]
