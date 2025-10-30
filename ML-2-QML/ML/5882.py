"""GraphQNNHybrid – classical implementation.

Provides:
- Random feed‑forward GNN with tanh activations.
- Classical Quanvolution filter (2×2 patch conv).
- Hybrid classifier that accepts either classical or quantum filter.
- Utility functions for fidelity graphs.
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
#  Classical GNN utilities
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
        for w in weights:
            current = torch.tanh(w @ current)
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
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
#  Classical quanvolution
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    """Simple 2×2 patch convolution followed by flattening."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    """Hybrid classifier that can switch between classical and quantum filter."""
    def __init__(self, use_quantum: bool = False) -> None:
        super().__init__()
        if use_quantum:
            from.qmodule_quantum import QuantumQuanvolutionFilter  # local import
            self.qfilter = QuantumQuanvolutionFilter()
        else:
            self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: Tensor) -> Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

# --------------------------------------------------------------------------- #
#  Combined dual‑branch model
# --------------------------------------------------------------------------- #
class GraphQNNHybrid(nn.Module):
    """Classical graph network + optional quanvolution classifier."""
    def __init__(self, graph_arch: Sequence[int], use_quantum_filter: bool = False) -> None:
        super().__init__()
        self.graph_arch = list(graph_arch)
        self.weights = [_random_linear(in_f, out_f) for in_f, out_f in zip(graph_arch[:-1], graph_arch[1:])]
        self.qclassifier = QuanvolutionClassifier(use_quantum=use_quantum_filter)

    def forward(self, graph_features: Tensor, image: Tensor) -> Tuple[Tensor, List[Tensor]]:
        # graph forward
        activations = [graph_features]
        current = graph_features
        for w in self.weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        # image classifier forward
        logits = self.qclassifier(image)
        return logits, activations

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "GraphQNNHybrid",
]
