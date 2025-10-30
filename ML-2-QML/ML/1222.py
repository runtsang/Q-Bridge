"""GraphQNN hybrid module – classical side.

This module extends the original GraphQNN utilities by adding a lightweight
graph neural network that learns node embeddings and an optional attention
mechanism.  The embeddings can be fed into the quantum module defined in the
QML counterpart.  The API keeps the original helper functions for
compatibility.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
#  Original helper functions – unchanged for backward compatibility
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
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
#  Hybrid GNN – new functionality
# --------------------------------------------------------------------------- #
class GraphQNN__gen165(nn.Module):
    """
    Lightweight GNN that learns node embeddings and optionally applies
    an attention mechanism.  The class is compatible with the original
    GraphQNN interface; its ``predict`` method returns a tensor of
    shape (num_nodes, out_features).
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        use_attention: bool = False,
    ):
        super().__init__()
        self.encoder = nn.Linear(in_features, hidden_dim)
        self.use_attention = use_attention
        if use_attention:
            self.attn = nn.Linear(hidden_dim, 1)
        self.decoder = nn.Linear(hidden_dim, out_features)

    def forward(self, graph: nx.Graph, node_features: Tensor) -> Tensor:
        """
        Forward pass that performs one message‑passing step followed by
        optional attention.
        """
        x = self.encoder(node_features)  # (N, hidden_dim)
        # Message passing: mean of neighbor embeddings
        new_x = torch.zeros_like(x)
        nodes = list(graph.nodes())
        idx_map = {n: i for i, n in enumerate(nodes)}
        for node in nodes:
            i = idx_map[node]
            neigh = list(graph.neighbors(node))
            if neigh:
                neigh_idx = torch.tensor([idx_map[n] for n in neigh], dtype=torch.long)
                neigh_emb = x[neigh_idx]
                new_x[i] = neigh_emb.mean(dim=0)
            else:
                new_x[i] = x[i]
        x = x + new_x  # residual connection
        if self.use_attention:
            attn = torch.sigmoid(self.attn(x))
            x = x * attn
        return x

    def predict(self, graph: nx.Graph, node_features: Tensor) -> Tensor:
        """
        Compute the final output by decoding the node embeddings.
        """
        embeddings = self.forward(graph, node_features)
        return self.decoder(embeddings)

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN__gen165",
]
