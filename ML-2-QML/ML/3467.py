"""Unified Graph‑QNN + Classical LSTM for graph‑structured sequential data.

This module implements a classical hybrid model that first
propagates a graph through a lightweight feed‑forward GNN
(`SimpleGNN`) to produce node embeddings, then feeds the
sequence of embeddings into a standard `nn.LSTM`.  The design mirrors
the original GraphQNN and QLSTM seeds while adding a graph‑to‑sequence
pipeline and a fidelity‑based regulariser that can be leveraged
in downstream experiments.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import List, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Classical utilities (mirrors GraphQNN seed)
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(
    weight: Tensor, samples: int
) -> List[Tuple[Tensor, Tensor]]:
    """Generate random training pairs for a given target weight."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1))
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(arch: Sequence[int], samples: int):
    """Produce a toy network of random linear layers and a training set."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(arch[:-1], arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target = weights[-1]
    training = random_training_data(target, samples)
    return list(arch), weights, training, target

def feedforward(
    arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]
) -> List[List[Tensor]]:
    """Forward pass through a list of linear layers, returning activations."""
    activations: List[List[Tensor]] = []
    for x, _ in samples:
        layerwise = [x]
        current = x
        for w in weights:
            current = torch.tanh(w @ current)
            layerwise.append(current)
        activations.append(layerwise)
    return activations

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two unit‑norm vectors."""
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
    """Build a weighted graph from pairwise state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# Simple graph neural network
# --------------------------------------------------------------------------- #
class SimpleGNN(nn.Module):
    """Feed‑forward GNN that transforms an adjacency row vector."""
    def __init__(self, arch: Sequence[int], device: torch.device | None = None):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:])]
        )
        self.device = device

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x

def graph_to_embeddings(graph: nx.Graph, gnn: SimpleGNN) -> Tensor:
    """Return a (N, out_dim) tensor of node embeddings for a fixed‑size graph."""
    nodes = list(graph.nodes)
    adj = nx.to_numpy_array(graph, nodelist=nodes, dtype=float)
    x = torch.tensor(adj, device=gnn.device, dtype=torch.float32)
    return gnn(x)

# --------------------------------------------------------------------------- #
# Unified hybrid model
# --------------------------------------------------------------------------- #
class UnifiedGraphQLSTM(nn.Module):
    """Classical hybrid model: GNN → LSTM → linear head."""
    def __init__(
        self,
        gnn_arch: Sequence[int],
        lstm_input_dim: int,
        lstm_hidden_dim: int,
        lstm_output_dim: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        if gnn_arch[0]!= lstm_input_dim:
            raise ValueError("gnn_arch[0] must equal lstm_input_dim")
        self.gnn = SimpleGNN(gnn_arch, device=device)
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_dim)
        self.final = nn.Linear(lstm_hidden_dim, lstm_output_dim)

    def forward(self, graph: nx.Graph) -> Tensor:
        """Run a graph through the GNN and LSTM, returning logits."""
        embeddings = graph_to_embeddings(graph, self.gnn)  # (N, H)
        seq = embeddings.unsqueeze(1)  # (N, 1, H)
        lstm_out, _ = self.lstm(seq)
        out = self.final(lstm_out.squeeze(1))
        return out

__all__ = [
    "UnifiedGraphQLSTM",
    "SimpleGNN",
    "graph_to_embeddings",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
