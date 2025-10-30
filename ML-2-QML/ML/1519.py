"""
Extended Graph Neural Network with attention and multi‑head message passing.
Provides a unified interface for classical and quantum back‑ends.
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

def _random_linear(in_features: int, out_features: int) -> nn.Linear:
    """Return a linear layer with random weights and biases."""
    linear = nn.Linear(in_features, out_features, bias=True)
    nn.init.xavier_uniform_(linear.weight)
    nn.init.zeros_(linear.bias)
    return linear

def random_training_data(weight: nn.Linear, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training data from a target linear layer."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.in_features, dtype=torch.float32)
        target = weight(features)
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random classical GNN with the given architecture."""
    weights: List[nn.Linear] = []
    for in_features, out_features in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_features, out_features))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return squared cosine similarity between two feature vectors."""
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

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[nn.Linear],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Forward propagation using a list of linear layers."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight(current))
            activations.append(current)
        stored.append(activations)
    return stored

class GraphQNN(nn.Module):
    """
    Classical graph neural network with multi‑head attention.
    """

    def __init__(
        self,
        arch: Sequence[int],
        num_heads: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.arch = list(arch)
        self.num_layers = len(arch) - 1
        self.num_heads = num_heads
        self.dropout = dropout

        # Linear layers for query, key, value and output per layer
        self.W_q = nn.ModuleList()
        self.W_k = nn.ModuleList()
        self.W_v = nn.ModuleList()
        self.W_out = nn.ModuleList()

        for i in range(self.num_layers):
            in_dim = arch[i]
            out_dim = arch[i + 1]
            self.W_q.append(nn.Linear(in_dim, out_dim))
            self.W_k.append(nn.Linear(in_dim, out_dim))
            self.W_v.append(nn.Linear(in_dim, out_dim))
            self.W_out.append(nn.Linear(out_dim, out_dim))

    def forward(self, G: nx.Graph, features: Tensor) -> List[Tensor]:
        """
        Forward pass through the network.
        :param G: NetworkX graph defining node connections.
        :param features: Tensor of shape (N, in_dim) with node features.
        :return: List of feature matrices, one per layer (including input).
        """
        h = features
        outputs = [h]

        # Adjacency mask (including self‑loops)
        adj = nx.to_numpy_array(G, dtype=float)
        mask = torch.tensor(adj > 0, dtype=torch.bool, device=features.device)
        mask.fill_diagonal_(True)

        for l in range(self.num_layers):
            Q = self.W_q[l](h)  # (N, out_dim)
            K = self.W_k[l](h)
            V = self.W_v[l](h)

            # Scaled dot‑product attention
            scores = torch.matmul(Q, K.transpose(0, 1)) / (K.shape[1] ** 0.5)  # (N, N)
            # Mask non‑neighbors
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = F.softmax(scores, dim=1)

            # Message aggregation
            h = torch.matmul(attn, V)

            if self.dropout > 0.0:
                h = F.dropout(h, self.dropout, training=self.training)

            # Linear output transformation
            h = self.W_out[l](h)
            outputs.append(h)

        return outputs

__all__ = [
    "GraphQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
