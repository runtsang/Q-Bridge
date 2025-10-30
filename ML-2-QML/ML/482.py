"""
GraphQNN__gen119.py

Hybrid classical‑quantum graph neural network utilities that extend the original
GraphQNN module.  The module keeps the original API (feedforward, fidelity_adjacency,
random_network, random_training_data, state_fidelity) while adding a
`GraphQNN` class that combines a classical embedding with a PennyLane
variational quantum circuit and a `GraphQNNTrainer` that can optimise the
model end‑to‑end.

The module is intentionally lightweight – it does not depend on heavy
training pipelines, but it exposes all the building blocks needed for
research experiments.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterable, Sequence
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1.  Core utilities (unchanged from the seed)
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(
    weight: torch.Tensor,
    samples: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate a synthetic dataset for a linear target transformation."""
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Build a list of random linear weights, synthetic training data and the
    target weight for the final layer."""
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
    """Return the activations for every layer of a purely classical feed‑forward network."""
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
    """Compute the squared overlap between two classical feature vectors."""
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
    """Construct a weighted graph from state fidelities."""
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
# 2.  Hybrid classical‑quantum architecture
# --------------------------------------------------------------------------- #

class GraphQNN(nn.Module):
    """
    Hybrid Graph Neural Network.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Architecture: list of hidden dimensions, e.g. [4, 8, 2].
    num_qubits : int, optional
        Number of qubits used in the variational layer.  Defaults to the
        last hidden dimension.
    learning_rate : float, optional
        Learning rate for the internal optimizer (used only when the module
        is trained via the provided ``train`` helper).
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        num_qubits: int | None = None,
        learning_rate: float = 0.01,
    ) -> None:
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.num_qubits = num_qubits or self.qnn_arch[-1]
        self.learning_rate = learning_rate

        # Classical embedding layers
        self.classical_layers = nn.ModuleList()
        for in_f, out_f in zip(self.qnn_arch[:-1], self.qnn_arch[1:]):
            self.classical_layers.append(nn.Linear(in_f, out_f))

        # Quantum variational layer
        self._setup_quantum_layer()

    def _setup_quantum_layer(self) -> None:
        """Instantiate a PennyLane qnode that acts on `self.num_qubits` qubits."""
        dev = qml.device("default.qubit", wires=self.num_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(x: torch.Tensor) -> torch.Tensor:
            # Encode the classical features
            for i, val in enumerate(x):
                qml.RY(val, wires=i)
            # Ansatz: a single layer of CNOT + Ry rotations
            for i in range(self.num_qubits):
                qml.RY(0.1 * torch.randn(1), wires=i)
                if i < self.num_qubits - 1:
                    qml.CNOT(wires=[i, i + 1])
            # Return expectation of PauliZ on all wires
            return qml.expval(qml.PauliZ(0))

        self.qnode = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid network."""
        # Classical path
        for layer in self.classical_layers:
            x = torch.tanh(layer(x))
        # Quantum path
        return self.qnode(x)

    def train(
        self,
        train_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        epochs: int = 10,
        fidelity_threshold: float = 0.8,
    ) -> List[nx.Graph]:
        """
        Simple training loop that optimises the network parameters using MSE loss.
        After each epoch a fidelity‑based graph is built from the current batch outputs.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        graphs: List[nx.Graph] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in train_loader:
                optimizer.zero_grad()
                pred = self.forward(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            # Build fidelity graph from the current batch outputs
            outputs = [self.forward(x).detach() for x, _ in train_loader]
            graph = fidelity_adjacency(outputs, fidelity_threshold)
            graphs.append(graph)
        return graphs

# --------------------------------------------------------------------------- #
# 3.  Trainer helper
# --------------------------------------------------------------------------- #

class GraphQNNTrainer:
    """
    Helper that orchestrates training of a hybrid GraphQNN against a dataset
    and records the evolution of the fidelity graph.

    Parameters
    ----------
    model : GraphQNN
        The hybrid network to optimise.
    optimizer : torch.optim.Optimizer
        Optimiser for the classical parameters.
    loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Loss function.
    fidelity_threshold : float
        Threshold for constructing the fidelity adjacency graph.
    """

    def __init__(
        self,
        model: nn.Module,
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
        pred = self.model(x)
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
            # Build graph from current batch outputs
            outputs = [self.model(x).detach() for x, _ in data_loader]
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
