"""GraphQNNWithAttention: Classical hybrid graph neural network with self‑attention.

This module extends the original GraphQNN by inserting a lightweight
self‑attention block at every propagation step.  All tensors are
PyTorch objects so the class can be dropped into a standard training
pipeline."""
import math
import itertools
from typing import Iterable, List, Sequence, Tuple

import torch
import networkx as nx
import numpy as np

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix with shape (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic (x, y) pairs where y = Wx."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        x = torch.randn(weight.size(1), dtype=torch.float32)
        y = weight @ x
        dataset.append((x, y))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random network, its training set and the target weight."""
    weights: List[Tensor] = [_random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two normalised vectors."""
    a_n = a / (torch.norm(a) + 1e-12)
    b_n = b / (torch.norm(b) + 1e-12)
    return float((a_n @ b_n).item() ** 2)

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Construct a weighted graph from state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

# --------------------------------------------------------------------------- #
# Self‑attention helper
# --------------------------------------------------------------------------- #

class _SelfAttention:
    """A lightweight transformer‑style attention block."""

    def __init__(self, dim: int):
        self.dim = dim
        # rotation and entangle matrices are small learnable parameters
        self.rotation = torch.randn(dim, dim, dtype=torch.float32)
        self.entangle = torch.randn(dim, dim, dtype=torch.float32)

    def forward(self, x: Tensor) -> Tensor:
        """Apply attention to a batch of vectors."""
        q = x @ self.rotation.t()
        k = x @ self.entangle.t()
        v = x
        scores = torch.softmax(q @ k.t() / math.sqrt(self.dim), dim=-1)
        return scores @ v

# --------------------------------------------------------------------------- #
# Main hybrid class
# --------------------------------------------------------------------------- #

class GraphQNNWithAttention:
    """
    Classical graph neural network that augments each linear layer with a
    self‑attention block.  The architecture is fully parameterised by
    ``arch`` – a sequence of layer widths.  The class exposes the same
    public API as the original GraphQNN module, but the forward pass
    interleaves attention after every linear transformation.
    """

    def __init__(self, arch: Sequence[int], attention_dim: int = 4):
        self.arch = list(arch)
        self.attention_dim = attention_dim
        self.weights = [_random_linear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:])]
        self.attention = _SelfAttention(attention_dim)

    # --------------------------------------------------------------------- #
    # Feed‑forward with attention
    # --------------------------------------------------------------------- #
    def feedforward(
        self,
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Return the list of activations for each sample."""
        all_activations: List[List[Tensor]] = []
        for x, _ in samples:
            activations: List[Tensor] = [x]
            current = x
            for w in self.weights:
                current = torch.tanh(w @ current.t()).t()
                current = self.attention.forward(current)
                activations.append(current)
            all_activations.append(activations)
        return all_activations

    # --------------------------------------------------------------------- #
    # Convenience helpers
    # --------------------------------------------------------------------- #
    def trainable_parameters(self) -> List[Tensor]:
        """Return all tensors that should be optimised."""
        return self.weights + [self.attention.rotation, self.attention.entangle]

    def state_fidelity(self, a: Tensor, b: Tensor) -> float:
        return state_fidelity(a, b)

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

__all__ = [
    "GraphQNNWithAttention",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
