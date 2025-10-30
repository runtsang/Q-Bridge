"""GraphQNNGen348: Classical ML implementation with attention‑augmented layers and graph-based fidelity adjacency."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
#  Core utilities – random network generation and training data
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Random weight matrix for a dense layer."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic input/target pairs using the target weight."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Sample a random feed‑forward network and its training data."""
    weights: List[Tensor] = [_random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

# --------------------------------------------------------------------------- #
#  Attention helper – classical self‑attention module
# --------------------------------------------------------------------------- #

class ClassicalSelfAttention:
    """Self‑attention block that outputs a weighted sum of the inputs."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        # Map inputs to query/key/value
        Q = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        K = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        V = torch.as_tensor(inputs, dtype=torch.float32)
        # Attention scores
        scores = torch.softmax(Q @ K.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ V).numpy()

# --------------------------------------------------------------------------- #
#  Feed‑forward with attention‑augmented linear layers
# --------------------------------------------------------------------------- #

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
    attention: ClassicalSelfAttention | None = None,
) -> List[List[Tensor]]:
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations: List[Tensor] = [features]
        current = features
        for layer, weight in enumerate(weights, start=1):
            # Attention modulation of the input
            if attention is not None:
                attn_out = attention.run(
                    rotation_params=weight.detach().numpy(),
                    entangle_params=weight.detach().numpy(),
                    inputs=features.numpy(),
                )
                current = torch.as_tensor(attn_out, dtype=torch.float32)
            # Linear transformation + non‑linearity
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

# --------------------------------------------------------------------------- #
#  Fidelity utilities – state overlap and graph construction
# --------------------------------------------------------------------------- #

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared inner‑product fidelity between two classical vectors."""
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
    """Create a weighted graph from pairwise state fidelities."""
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
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "ClassicalSelfAttention",
]
