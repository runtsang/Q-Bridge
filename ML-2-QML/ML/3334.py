"""Hybrid classical graph neural network with optional self‑attention.

The module exposes ``GraphQNNHybrid`` which operates in a fully classical
mode using PyTorch for the feed‑forward pass and NetworkX for graph
construction.  The classical self‑attention block is implemented with
dot‑product attention and can be toggled on or off at initialization.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional

import networkx as nx
import numpy as np
import torch

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# Classical self‑attention helper
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention:
    """Dot‑product self‑attention implemented with PyTorch."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1),
                                dtype=torch.float32)
        key   = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1),
                                dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


# --------------------------------------------------------------------------- #
# Core utilities
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target   = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(qnn_arch: Sequence[int],
                weights: Sequence[Tensor],
                samples: Iterable[Tuple[Tensor, Tensor]],
                attention: Optional[ClassicalSelfAttention] = None) -> List[List[Tensor]]:
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        if attention is not None:
            # Apply self‑attention to the raw input before the first layer
            attn_out = attention.run(np.random.randn(4, 4),
                                     np.random.randn(4, 4),
                                     features.numpy())
            current = torch.as_tensor(attn_out, dtype=torch.float32)
            activations.append(current)
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(states: Sequence[Tensor],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
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
# Hybrid class
# --------------------------------------------------------------------------- #
class GraphQNNHybrid:
    """Hybrid classical graph‑neural‑network with optional self‑attention."""

    def __init__(self,
                 arch: Sequence[int],
                 use_attention: bool = False,
                 attention_type: str = "classical"):
        self.arch = list(arch)
        self.weights: List[Tensor] = []
        self.use_attention = use_attention
        if use_attention:
            if attention_type == "classical":
                self.attention = ClassicalSelfAttention(embed_dim=4)
            else:
                raise ValueError("Quantum attention not available in classical mode")
        else:
            self.attention = None

    def random_initialize(self, samples: int = 10):
        _, self.weights, _, _ = random_network(self.arch, samples)

    def feedforward(self,
                    samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        return feedforward(self.arch, self.weights, samples, attention=self.attention)

    def fidelity_graph(self,
                       threshold: float,
                       *,
                       secondary: float | None = None) -> nx.Graph:
        # Execute a forward pass once to collect final layer activations
        states = [layer[-1] for layer in self.feedforward(self.samples)]
        return fidelity_adjacency(states, threshold, secondary=secondary)


__all__ = [
    "GraphQNNHybrid",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
