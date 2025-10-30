"""Hybrid classical kernel for graph data, combining RBF, graph fidelity, and optional quantum weighting."""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
import torch
from torch import nn

__all__ = [
    "HybridGraphKernel",
    "kernel_matrix",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
]

class HybridGraphKernel(nn.Module):
    """Hybrid classical kernel combining RBF similarity and fidelity-based graph weighting."""
    def __init__(self, rbf_gamma: float = 1.0, graph_threshold: float = 0.8, quantum_weight: float = 1.0):
        super().__init__()
        self.rbf_gamma = rbf_gamma
        self.graph_threshold = graph_threshold
        self.quantum_weight = quantum_weight

    def rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel between two embeddings."""
        diff = x - y
        return torch.exp(-self.rbf_gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def fidelity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute classical state fidelity (squared overlap)."""
        a = x / (torch.norm(x) + 1e-12)
        b = y / (torch.norm(y) + 1e-12)
        return (torch.dot(a, b)**2)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return hybrid kernel value."""
        rbf_val = self.rbf(x, y)
        fid_val = self.fidelity(x, y)
        return self.quantum_weight * rbf_val * fid_val

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], rbf_gamma: float = 1.0, quantum_weight: float = 1.0) -> np.ndarray:
    """Compute Gram matrix between two sets of node embeddings."""
    kernel = HybridGraphKernel(rbf_gamma=rbf_gamma, quantum_weight=quantum_weight)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = (
            state_i / (torch.norm(state_i) + 1e-12)
        ).dot(state_j / (torch.norm(state_j) + 1e-12)) ** 2
        fid = fid.item()
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    dataset: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    weights = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight
