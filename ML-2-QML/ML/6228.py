"""Hybrid GraphQNN: classical backbone + quantum variational layer.

This module extends the original GraphQNN interface by adding a learnable
graph‑embedding layer and a Pennylane variational circuit.  The forward
pass returns both the classical activations and the quantum state vector
so that users can compute fidelity graphs or use the quantum output as a
feature.  The design keeps the original `feedforward` and
`fidelity_adjacency` helpers unchanged, while exposing a new
`GraphQNN` class that can be trained end‑to‑end.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import networkx as nx
import pennylane as qml
import pennylane.numpy as np

Tensor = torch.Tensor
QDArray = np.ndarray

# --------------------------------------------------------------------------- #
#  Helper functions (unchanged from the seed)
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a weight matrix of shape (out, input) with normal entries."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> list[tuple[Tensor, Tensor]]:
    """Generate random feature/target pairs for a single weight matrix."""
    dataset = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Return architecture, classical weights, training data and target weight."""
    weights = [_random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two classical vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5
) -> nx.Graph:
    """Create a weighted graph from pairwise fidelities."""
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
#  Hybrid GraphQNN model
# --------------------------------------------------------------------------- #
class GraphQNN(nn.Module):
    """Hybrid classical‑quantum graph neural network.

    The model consists of a classical feed‑forward backbone followed by a
    Pennylane variational circuit.  The classical layers produce an embedding
    that is fed into the quantum circuit as rotation angles.
    """

    def __init__(self, arch: Sequence[int], quantum_device: str = "default.qubit", device: str | None = None):
        super().__init__()
        self.arch = list(arch)
        self.device = torch.device(device or "cpu")
        self.quantum_device = quantum_device

        # Classical linear layers
        self.classical_layers = nn.ModuleList()
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            self.classical_layers.append(nn.Linear(in_f, out_f))

        # Quantum variational parameters
        self.n_qubits = self.arch[-1]
        self.qlayer = nn.Parameter(torch.randn(self.n_qubits, requires_grad=True))

        # Pennylane device and qnode
        self.dev = qml.device(self.quantum_device, wires=self.n_qubits)
        self.qnode = qml.QNode(self._quantum_circuit, self.dev, interface="torch")

    def _quantum_circuit(self, x: Tensor) -> QDArray:
        # Apply classical embedding as rotation angles
        for i in range(self.n_qubits):
            qml.RY(x[i], wires=i)
        # Entangling layer
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        # Parameterized rotations
        for i in range(self.n_qubits):
            qml.RZ(self.qlayer[i], wires=i)
        return qml.state()

    def forward(self, x: Tensor) -> tuple[Tensor, QDArray]:
        # Classical forward pass
        h = x
        for layer in self.classical_layers:
            h = torch.tanh(layer(h))
        # Quantum forward pass
        qstate = self.qnode(h)
        return h, qstate

    def get_classical_weights(self) -> list[Tensor]:
        return [layer.weight for layer in self.classical_layers]

    def get_quantum_weights(self) -> Tensor:
        return self.qlayer

__all__ = [
    "GraphQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
