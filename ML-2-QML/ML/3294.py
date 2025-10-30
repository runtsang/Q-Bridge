"""Hybrid Graph Neural Network combining classical feedforward and RBF kernel embeddings.

This module expands the original GraphQNN utilities by adding a kernel-based
graph construction step and a lightweight RBF kernel module.  The
class `HybridGraphQNN` can be used in pure‑Python or PyTorch pipelines
and mirrors the API of the quantum counterpart for seamless
interchangeability.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple

import numpy as np
import torch
from torch import nn
import networkx as nx

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Random data generation utilities
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Random weight matrix with shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training pairs (x, Wx) for a given linear map."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random classical network and synthetic data for its last layer."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

# --------------------------------------------------------------------------- #
# Core feedforward and fidelity utilities
# --------------------------------------------------------------------------- #

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Compute activations for each sample across all layers."""
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
    """Normalized squared overlap between two vectors."""
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
    """Build a weighted graph from state fidelities."""
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
# RBF kernel utilities
# --------------------------------------------------------------------------- #

class KernalAnsatz(nn.Module):
    """RBF kernel implementation compatible with the quantum interface."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper that exposes a single forward method for two tensors."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[Tensor], b: Sequence[Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute the Gram matrix between two collections of vectors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Hybrid graph neural network module
# --------------------------------------------------------------------------- #

class HybridGraphQNN(nn.Module):
    """
    A lightweight hybrid graph neural network that couples a classical
    feedforward backbone with an RBF kernel‑based graph construction.
    The API mirrors the quantum counterpart so the same class name can
    be imported from either the classical or quantum module.
    """
    def __init__(self, arch: Sequence[int], gamma: float = 1.0) -> None:
        super().__init__()
        self.arch = list(arch)
        self.gamma = gamma
        self.weights = nn.ParameterList(
            [nn.Parameter(_random_linear(in_f, out_f))
             for in_f, out_f in zip(self.arch[:-1], self.arch[1:])]
        )
        self.kernel = Kernel(gamma)

    def forward(self, x: Tensor) -> List[Tensor]:
        """Return activations for a single input vector."""
        activations = [x]
        current = x
        for weight in self.weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        return activations

    def build_graph(self, states: Sequence[Tensor], threshold: float) -> nx.Graph:
        """Construct a graph from node embeddings using kernel fidelity."""
        return fidelity_adjacency(states, threshold)

    def compute_kernel_matrix(self, a: Sequence[Tensor], b: Sequence[Tensor]) -> np.ndarray:
        """Return the Gram matrix between two sets of node embeddings."""
        return kernel_matrix(a, b, self.gamma)

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "kernel_matrix",
    "HybridGraphQNN",
]
