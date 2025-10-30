"""
GraphQNNGen607 – Classical implementation.

This module keeps the original GraphQNN utilities but augments each layer
with a self‑attention block.  The network can be instantiated, trained
and evaluated entirely on a CPU using PyTorch tensors.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import torch
import networkx as nx

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
#  Utility helpers
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix for a dense layer."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate (x, Wx) pairs for supervised training."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


# --------------------------------------------------------------------------- #
#  Attention block
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(torch.nn.Module):
    """
    Simple self‑attention module that mirrors the quantum block.
    Parameters are learned during training.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = torch.nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.softmax(q @ k.t() / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v


# --------------------------------------------------------------------------- #
#  Graph‑based network
# --------------------------------------------------------------------------- #
def random_network(qnn_arch: Sequence[int], samples: int):
    """
    Build a random classical network with attention layers.
    Returns architecture, weights, training data and target weight.
    """
    weights: List[Tensor] = []
    attentions: List[ClassicalSelfAttention] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
        attentions.append(ClassicalSelfAttention(embed_dim=out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, attentions, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    attentions: Sequence[ClassicalSelfAttention],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """
    Forward pass through the network; each layer applies a linear
    transformation followed by self‑attention.
    """
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight, attention in zip(weights, attentions):
            current = torch.tanh(weight @ current)
            current = attention(current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two classical states."""
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


__all__ = [
    "ClassicalSelfAttention",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
