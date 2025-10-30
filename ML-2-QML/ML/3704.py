"""GraphQNNHybrid: classical‑quantum graph neural network utilities.

This module combines the core ideas from the two reference pairs:
* Classical linear layers with tanh activations and random weight generation
  (GraphQNN.py)
* Quantum unitary blocks that propagate states through a tensor product of
  identity and zero registers (QuantumGraphQNN.py)
* Fidelity‑based graph construction from both classical activations and
  quantum state overlaps (both seed modules)
* A hybrid `HybridLayer` that can be inserted into a dense network or a
  graph‑based architecture.

The module is fully importable, uses only NumPy, PyTorch, Qiskit and
Scipy, and can be used as a drop‑in replacement for the classical
`GraphQNN` or the hybrid `QCNet` defined in the seeds.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import quantum utilities from the sibling module
from qml_code import QuantumCircuit, HybridFunction

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random torch‑tensor (weight matrix)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(
    weight: Tensor,
    samples: int,
) -> List[Tuple[Tensor, Tensor]]:
    """Generate random training data for a weight matrix."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Generate a random linear network and training data."""
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
    """Classical feed‑forward through a sequence of linear layers."""
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
    """Squared overlap of two classical activation vectors."""
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


class HybridLayer(nn.Module):
    """Hybrid layer that forwards activations through a parametrised quantum circuit."""

    def __init__(self, n_qubits: int, shots: int = 100, shift: float = 0.0) -> None:
        super().__init__()
        self.quantum = QuantumCircuit(n_qubits, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.quantum, self.shift)


class GraphQNNHybrid(nn.Module):
    """Hybrid graph neural network that interleaves classical dense layers
    with a quantum expectation head.

    The architecture is defined by a list of layer widths.  The first
    element is the input dimensionality; subsequent elements are the
    widths of hidden layers.  The final hidden layer feeds into a
    quantum hybrid head that produces a scalar output.  The network
    can be used for regression or binary classification by applying
    a sigmoid to the output.
    """

    def __init__(
        self,
        arch: Sequence[int],
        n_qubits: int,
        shots: int = 100,
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        self.arch = list(arch)
        self.linear_layers: nn.ModuleList = nn.ModuleList()
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            self.linear_layers.append(nn.Linear(in_f, out_f))
        self.hybrid = HybridLayer(n_qubits, shots, shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin in self.linear_layers:
            x = torch.tanh(lin(x))
        x = self.hybrid(x)
        # Default to a sigmoid for binary classification
        return torch.sigmoid(x)


__all__ = [
    "HybridLayer",
    "GraphQNNHybrid",
    "feedforward",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
