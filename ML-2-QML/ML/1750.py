"""
GraphQNNHybrid.py (classical)

A hybrid graph neural network that extends the original seed by adding a residual GAT backbone and a feed‑forward sequence.  Utility functions for synthetic data generation, state fidelity, and fidelity‑based adjacency graphs are included.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> nn.Parameter:
    """Return a trainable weight matrix with normal initialization."""
    return nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float32))

def random_training_data(
    weight: Tensor, samples: int
) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic data for a linear target weight."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(
    qnn_arch: Sequence[int], samples: int
) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Create a random classical network with matching architecture."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]
) -> List[List[Tensor]]:
    """Run a forward pass for each sample, recording activations."""
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
    """Cosine similarity squared, robust against zero norms."""
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
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class GraphQNNHybrid(nn.Module):
    """Hybrid graph neural network with a residual GAT backbone and feed‑forward layers."""
    def __init__(self, arch: Sequence[int], heads: int = 4):
        super().__init__()
        self.arch = arch
        self.gat = GATConv(in_channels=arch[0], out_channels=arch[1], heads=heads, concat=False)
        self.residual = nn.Linear(arch[1], arch[1])
        self.linear_layers = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:])])

    def forward(self, data: Data) -> List[Tensor]:
        x, edge_index = data.x, data.edge_index
        x = self.gat(x, edge_index)
        x = F.relu(self.residual(x) + x)
        activations = [x]
        for layer in self.linear_layers:
            x = torch.tanh(layer(x))
            activations.append(x)
        return activations
