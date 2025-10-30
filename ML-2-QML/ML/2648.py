"""HybridSelfAttentionGraphQNN: classical self‑attention + graph utilities."""
from __future__ import annotations

import itertools
import numpy as np
import torch
import networkx as nx
from typing import Iterable, Sequence, List, Tuple

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Classical self‑attention building block
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention:
    """Differentiable self‑attention module with optional bias."""
    def __init__(self, embed_dim: int, bias: bool = True):
        self.embed_dim = embed_dim
        self.bias = bias
        self.query = torch.nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.key   = torch.nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.value = torch.nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Compute attention scores and return the weighted sum of values.
        Parameters
        ----------
        inputs : torch.Tensor
            shape (batch, seq_len, embed_dim)
        Returns
        -------
        output : torch.Tensor
            shape (batch, seq_len, embed_dim)
        """
        q = torch.einsum('bsi,ij->bsh', inputs, self.query)
        k = torch.einsum('bsi,ij->bsh', inputs, self.key)
        v = torch.einsum('bsi,ij->bsh', inputs, self.value)
        scores = torch.softmax(
            torch.einsum('bsh,bth->bsq', q, k) / np.sqrt(self.embed_dim), dim=-1
        )
        return torch.einsum('bsq,bsh->bsh', scores, v)

# --------------------------------------------------------------------------- #
# Graph utilities (classical)
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
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
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
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
# Hybrid orchestrator
# --------------------------------------------------------------------------- #
class HybridSelfAttentionGraphQNN:
    """Top‑level orchestrator that combines classical self‑attention with graph propagation."""
    def __init__(self, embed_dim: int, qnn_arch: Sequence[int]):
        self.attention = ClassicalSelfAttention(embed_dim)
        self.qnn_arch = list(qnn_arch)
        self.weights, self.training_data, self.target = random_network(qnn_arch, samples=10)[1:4]

    def run_attention(self, inputs: Tensor) -> Tensor:
        return self.attention.forward(inputs)

    def run_graph(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        return feedforward(self.qnn_arch, self.weights, samples)

    def build_fidelity_graph(self, states: Sequence[Tensor], threshold: float) -> nx.Graph:
        return fidelity_adjacency(states, threshold)

__all__ = ["HybridSelfAttentionGraphQNN", "ClassicalSelfAttention", "random_network",
           "random_training_data", "feedforward", "state_fidelity", "fidelity_adjacency"]
