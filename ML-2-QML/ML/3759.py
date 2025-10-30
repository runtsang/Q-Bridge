"""
Hybrid classical self‑attention with a graph‑based feed‑forward network.

The class exposes a ``run`` API that takes an input tensor, dummy
rotation and entangle parameters (for API parity with the quantum
seed) and returns the attention‑weighted representation.  Internally
it uses a standard PyTorch self‑attention block and a linear
feed‑forward network that mirrors the layer structure of the quantum
graph network from the seed.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import torch
import networkx as nx

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(
    weight: Tensor, samples: int
) -> List[Tuple[Tensor, Tensor]]:
    """Generate a dataset that targets a linear mapping defined by *weight*."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """
    Build a *random* classical network with the same shape as the
    quantum‑graph architecture.  The final weight is used as a target
    for the training data.
    """
    weights: List[Tensor] = []
    for in_features, out_features in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_features, out_features))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Classic feed‑forward through a sequence of linear + tanh layers."""
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
    """Return the squared‑overlap between two 1‑D tensors."""
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
    """Build a graph where each node is a state and edges encode fidelity‑based similarity."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class QuantumSelfAttentionGraphNet:
    """
    Hybrid classical self‑attention + feed‑forward network.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    qnn_arch : Sequence[int]
        Architecture of the linear layers that mimic the quantum graph.
    """

    def __init__(self, embed_dim: int, qnn_arch: Sequence[int]):
        self.embed_dim = embed_dim
        self.qnn_arch = list(qnn_arch)
        # Randomly initialise the linear attention weights
        self.query_weight = torch.nn.Parameter(
            torch.randn(self.embed_dim, self.embed_dim, dtype=torch.float32)
        )
        self.key_weight = torch.nn.Parameter(
            torch.randn(self.embed_dim, self.embed_dim, dtype=torch.float32)
        )
        # Linear layers for the feed‑forward path
        self.ff_weights: List[torch.nn.Parameter] = []
        for in_f, out_f in zip(self.qnn_arch[:-1], self.qnn_arch[1:]):
            w = torch.nn.Parameter(
                torch.randn(out_f, in_f, dtype=torch.float32)
            )
            self.ff_weights.append(w)

    def attention(self, inputs: Tensor) -> Tensor:
        """Compute a classical scaled dot‑product attention."""
        Q = inputs @ self.query_weight
        K = inputs @ self.key_weight
        V = inputs
        scores = torch.softmax(Q @ K.T / math.sqrt(self.embed_dim), dim=-1)
        return scores @ V

    def feedforward(self, x: Tensor) -> Tensor:
        """Linear + tanh feed‑forward path."""
        out = x
        for w in self.ff_weights:
            out = torch.tanh(w @ out)
        return out

    def run(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> np.ndarray:
        """
        Accepts a batch of input embeddings and dummy parameters
        (mirroring the quantum interface).  The parameters are ignored
        by the classical implementation but kept for API parity.
        """
        x = torch.as_tensor(inputs, dtype=torch.float32)
        attn_out = self.attention(x)
        ff_out = self.feedforward(attn_out)
        return ff_out.detach().numpy()

    def compute_adjacency(self, states: Sequence[np.ndarray], threshold: float) -> nx.Graph:
        """
        Build a graph of state fidelities from a list of 1‑D NumPy arrays.
        """
        torch_states = [torch.as_tensor(s, dtype=torch.float32) for s in states]
        return fidelity_adjacency(torch_states, threshold)

__all__ = ["QuantumSelfAttentionGraphNet"]
