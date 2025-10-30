"""Hybrid Graph Neural Network for classical & quantum experiments.

This module extends the original GraphQNN by adding a learnable adjacency
matrix and a simple two‑layer GCN.  It also exposes a hybrid loss that
combines mean‑squared‑error with a placeholder quantum fidelity term.
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

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix with shape (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate random feature‑target pairs for a simple linear network."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(
    qnn_arch: Sequence[int],
    samples: int,
) -> tuple[list[int], list[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Create a synthetic random network and training data."""
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
    """Forward pass through a pure‑linear network."""
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
    """Compute squared overlap between two state vectors."""
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

class GraphQNN__gen282(nn.Module):
    """
    Two‑layer Graph Convolutional Network with a learnable adjacency matrix.
    """

    def __init__(self, arch: Sequence[int], use_hybrid: bool = False):
        super().__init__()
        self.arch = list(arch)
        self.use_hybrid = use_hybrid
        # Learnable adjacency
        self.adj = nn.Parameter(torch.rand(self.arch[0], self.arch[0]))
        # Linear layers
        self.linear1 = nn.Linear(self.arch[0], self.arch[1])
        self.linear2 = nn.Linear(self.arch[1], self.arch[2])

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, n_features)
        # First layer
        h1 = torch.tanh(self.linear1(x))
        h1 = torch.matmul(self.adj, h1)  # adjacency weighting
        # Second layer
        h2 = torch.tanh(self.linear2(h1))
        return h2

    def train_step(
        self,
        data_loader,
        optimizer,
        criterion,
    ) -> float:
        """Single epoch training step with optional hybrid loss."""
        self.train()
        total_loss = 0.0
        for batch in data_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            outputs = self.forward(inputs)
            loss = criterion(outputs, targets)
            # Placeholder for quantum fidelity term
            if self.use_hybrid:
                fid = state_fidelity(outputs.flatten(), targets.flatten())
                loss += (1 - fid) * 0.1  # small weight
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data_loader)

__all__ = [
    "GraphQNN__gen282",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
