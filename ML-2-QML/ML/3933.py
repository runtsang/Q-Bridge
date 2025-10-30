"""Integrated classical Graph‑Quanvolution neural network.

This module combines the original GraphQNN utilities with a
classical quanvolution filter.  The architecture is identical to
the quantum counterpart in terms of graph connectivity but uses
pure PyTorch tensors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools
from typing import List, Tuple, Sequence, Iterable

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Generate a random weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Create synthetic training pairs for a linear layer."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Generate random weight matrices for a multi‑layer linear network."""
    weights: List[Tensor] = [_random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]):
    """Forward pass through the linear graph network."""
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
    """Squared overlap between two state vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                       *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class QuanvolutionFilter(nn.Module):
    """2‑D convolution followed by flattening, mimicking the quantum filter."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    def forward(self, x: Tensor) -> Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class IntegratedGraphQuanvolutionQNN(nn.Module):
    """Hybrid classical Graph‑Quanvolution neural network."""
    def __init__(self, qnn_arch: Sequence[int], in_channels: int = 1):
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.qfilter = QuanvolutionFilter(in_channels)
        # Compute flattened feature size
        dummy_input = torch.zeros(1, in_channels, 28, 28)
        feat_dim = self.qfilter(dummy_input).shape[1]
        self.linear = nn.Linear(feat_dim, self.qnn_arch[0])
        _, self.weights, _, _ = random_network(self.qnn_arch, samples=10)
    def forward(self, x: Tensor) -> Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        activations = [logits]
        current = logits
        for weight in self.weights[1:]:
            current = torch.tanh(weight @ current)
            activations.append(current)
        return activations[-1]
    def get_fidelity_graph(self, states: Sequence[Tensor], threshold: float,
                           *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

__all__ = [
    "IntegratedGraphQuanvolutionQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "QuanvolutionFilter",
]
