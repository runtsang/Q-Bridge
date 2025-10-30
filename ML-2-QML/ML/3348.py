"""Hybrid graph neural network classifier combining classical GNN and quantum-inspired structure."""

from __future__ import annotations

import itertools
from typing import Iterable, List, Tuple

import networkx as nx
import torch
import torch.nn as nn

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix for a linear layer."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training data from a target weight matrix."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: List[int], samples: int):
    """Create a random fully‑connected network and training data."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: List[int],
    weights: List[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Forward pass through a fully‑connected network."""
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
    """Squared overlap of two state vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: List[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a graph where edges denote fidelity above a threshold."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class HybridGraphClassifier(nn.Module):
    """Graph neural network classifier with quantum‑inspired layer structure."""
    def __init__(self, qnn_arch: List[int], adjacency: nx.Graph):
        super().__init__()
        self.qnn_arch = qnn_arch
        self.adjacency = adjacency
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f))
        self.activation = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        # Simple message passing: aggregate neighbor features
        for layer in self.layers[:-1]:
            agg = torch.zeros_like(x)
            for i in range(x.size(0)):
                neighbors = list(self.adjacency.neighbors(i))
                if neighbors:
                    agg[i] = x[neighbors].mean(dim=0)
                else:
                    agg[i] = x[i]
            x = self.activation(layer(agg))
        logits = self.layers[-1](x)
        return logits

def build_classifier_graph(num_nodes: int, depth: int) -> Tuple[HybridGraphClassifier, nx.Graph, List[int], Tensor]:
    """Create a random graph GNN classifier and its metadata."""
    # Architecture: input features = num_nodes, hidden layers = depth-1, output=2
    qnn_arch = [num_nodes] + [num_nodes] * (depth - 1) + [2]
    arch, weights, training_data, target_weight = random_network(qnn_arch, samples=0)
    # Generate adjacency from random states
    states = [torch.randn(num_nodes) for _ in range(num_nodes)]
    adjacency = fidelity_adjacency(states, threshold=0.8, secondary=0.5)
    model = HybridGraphClassifier(arch, adjacency)
    weight_sizes = [w.numel() for w in weights]
    return model, adjacency, weight_sizes, target_weight
