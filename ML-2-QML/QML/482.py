"""
GraphQNN__gen119.py

Quantum‑centric counterpart to the hybrid GraphQNN module.  The file keeps
the original utility functions (feedforward, fidelity_adjacency,
random_network, random_training_data, state_fidelity) while adding a
`GraphQNN` class that implements a fully quantum variational neural
network using PennyLane.  A `GraphQNNTrainer` helper is also provided
for end‑to‑end optimisation with back‑propagation through the quantum
circuit.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Any, List, Tuple

import networkx as nx
import pennylane as qml
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# 1.  Core utilities (unchanged from the seed)
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(
    weight: torch.Tensor,
    samples: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    stored: List[List[torch.Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
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
# 2.  Fully quantum architecture
# --------------------------------------------------------------------------- #

class GraphQNN:
    """
    Fully quantum Graph Neural Network implemented with PennyLane.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Architecture: list of hidden dimensions, e.g. [4, 8, 2].
    """

    def __init__(self, qnn_arch: Sequence[int]) -> None:
        self.qnn_arch = list(qnn_arch)
        self.num_qubits = self.qnn_arch[-1]
        self._setup_qnode()

    def _setup_qnode(self) -> None:
        dev = qml.device("default.qubit", wires=self.num_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(x: torch.Tensor) -> torch.Tensor:
            # Encode the input as rotation angles
            for i, val in enumerate(x):
                qml.RY(val, wires=i)
            # Variational ansatz: single layer of Ry rotations and CNOTs
            for i in range(self.num_qubits):
                qml.RY(0.1 * torch.randn(1), wires=i)
                if i < self.num_qubits - 1:
                    qml.CNOT(wires=[i, i + 1])
            # Output expectation of PauliZ on the last qubit
            return qml.expval(qml.PauliZ(self.num_qubits - 1))

        self.qnode = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the expectation value of the variational circuit."""
        return self.qnode(x)

# --------------------------------------------------------------------------- #
# 3.  Trainer helper
# --------------------------------------------------------------------------- #

class GraphQNNTrainer:
    """
    Optimiser for a purely quantum GraphQNN.

    Parameters
    ----------
    model : GraphQNN
        The quantum model to optimise.
    optimizer : torch.optim.Optimizer
        Optimiser for the circuit parameters.
    loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Loss function.
    fidelity_threshold : float
        Threshold for constructing the fidelity adjacency graph.
    """

    def __init__(
        self,
        model: GraphQNN,
        optimizer: torch.optim.Optimizer,
        loss_fn: Any,
        fidelity_threshold: float = 0.8,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.fidelity_threshold = fidelity_threshold
        self.graph_history: List[nx.Graph] = []

    def step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Perform a single optimisation step."""
        self.optimizer.zero_grad()
        pred = self.model.forward(x)
        loss = self.loss_fn(pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]], epochs: int = 10) -> None:
        """Train over the dataset and record fidelity graphs."""
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in data_loader:
                loss = self.step(x, y)
                epoch_loss += loss
            avg_loss = epoch_loss / len(data_loader)
            outputs = [self.model.forward(x).detach() for x, _ in data_loader]
            graph = fidelity_adjacency(outputs, self.fidelity_threshold)
            self.graph_history.append(graph)

# --------------------------------------------------------------------------- #
# 4.  Public interface
# --------------------------------------------------------------------------- #

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN",
    "GraphQNNTrainer",
]
