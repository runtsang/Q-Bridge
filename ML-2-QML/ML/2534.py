"""UnifiedSelfAttentionQNN – classical component.

The module defines a hybrid self‑attention mechanism that operates on
quantum state vectors.  It uses a small feed‑forward network to embed
each state into a classical vector space, builds a fidelity‑based graph,
and applies a masked self‑attention block.  The design is inspired by
the classical SelfAttention and GraphQNN utilities.
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
# Helper: small feed‑forward network for embedding
# --------------------------------------------------------------------------- #
def _make_ffn(in_dim: int, out_dim: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, out_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(out_dim, out_dim),
    )

# --------------------------------------------------------------------------- #
# Classical self‑attention block
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention:
    """Classic self‑attention over a list of quantum state vectors."""

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim
        # Each state is embedded from a scalar (norm) to embed_dim
        self._embed = _make_ffn(in_dim=1, out_dim=embed_dim)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: List[Tensor],
    ) -> Tensor:
        # Embed each input state by its norm
        embedded = torch.stack(
            [self._embed(inp.norm().unsqueeze(0)).squeeze(0) for inp in inputs],
            dim=0,
        )  # shape: (N, embed_dim)

        # Build a fidelity‑based graph from the embedded states
        graph = fidelity_adjacency(
            states=[row for row in embedded],
            threshold=0.8,
            secondary=0.6,
            secondary_weight=0.3,
        )

        # Create a boolean mask from the graph adjacency
        N = embedded.shape[0]
        mask = torch.zeros((N, N), dtype=torch.bool)
        for i, j in graph.edges():
            mask[i, j] = True
            mask[j, i] = True

        # Standard scaled dot‑product attention
        query = embedded
        key = embedded
        value = embedded
        scores = torch.softmax(
            query @ key.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1
        )

        # Apply the graph mask
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        scores = torch.softmax(scores, dim=-1)

        # Weighted sum over the sequence dimension
        output = (scores @ value).mean(dim=0)  # shape: (embed_dim,)
        return output

# --------------------------------------------------------------------------- #
# Graph utilities (inspired by GraphQNN)
# --------------------------------------------------------------------------- #
def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Compute squared overlap between two 1‑D tensors."""
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
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# Random network utilities (inspired by GraphQNN)
# --------------------------------------------------------------------------- #
def random_network(qnn_arch: Sequence[int], samples: int = 10):
    """Generate a random linear network and training data."""
    weights = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(torch.randn(out_f, in_f))
    target_weight = weights[-1]
    training_data = []
    for _ in range(samples):
        features = torch.randn(target_weight.size(1))
        target = target_weight @ features
        training_data.append((features, target))
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    stored = []
    for features, _ in samples:
        activations = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        stored.append(activations)
    return stored

# --------------------------------------------------------------------------- #
# UnifiedSelfAttentionQNN class
# --------------------------------------------------------------------------- #
class UnifiedSelfAttentionQNN:
    """Hybrid self‑attention that bridges classical and quantum representations.

    The class can be instantiated with an embedding dimension and an optional
    QNN architecture.  The `run` method accepts rotation and entanglement
    parameters (to be used by a quantum backend) and a list of quantum state
    vectors.  The states are embedded, a fidelity‑based graph is built,
    and a masked self‑attention is applied to produce a classical output.
    """

    def __init__(self, embed_dim: int = 4, qnn_arch: Optional[Sequence[int]] = None):
        self.embed_dim = embed_dim
        self.attention = ClassicalSelfAttention(embed_dim=embed_dim)
        self.qnn_arch = qnn_arch or [4, 4, 4]

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: List[Tensor],
    ) -> Tensor:
        """Run the hybrid attention block."""
        return self.attention.run(rotation_params, entangle_params, inputs)

# --------------------------------------------------------------------------- #
# Compatibility wrapper
# --------------------------------------------------------------------------- #
def SelfAttention() -> UnifiedSelfAttentionQNN:
    """Return a pre‑configured UnifiedSelfAttentionQNN instance."""
    return UnifiedSelfAttentionQNN(embed_dim=4)

__all__ = [
    "UnifiedSelfAttentionQNN",
    "SelfAttention",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
