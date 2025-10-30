"""
Quantum‑based Graph Neural Network with a variational circuit.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import pennylane as qml
import pennylane.numpy as np
import torch
import torch.nn as nn

Tensor = torch.Tensor

def random_unitary(num_qubits: int) -> Tensor:
    dim = 2 ** num_qubits
    mat = qml.math.random_unitary(dim)
    return torch.tensor(mat, dtype=torch.complex64)

def random_training_data(unitary: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    dim = unitary.shape[0]
    for _ in range(samples):
        state = torch.randn(dim, dtype=torch.complex64)
        state /= torch.norm(state)
        target = unitary @ state
        dataset.append((state, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    num_qubits = qnn_arch[-1]
    target_unitary = random_unitary(num_qubits)
    training_data = random_training_data(target_unitary, samples)
    params = torch.zeros(num_qubits, requires_grad=True)
    return list(qnn_arch), params, training_data, target_unitary

def feedforward(
    qnn_arch: Sequence[int],
    params: Tensor,
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[Tensor]:
    num_qubits = qnn_arch[-1]
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(state: Tensor) -> Tensor:
        qml.QubitStateVector(state, wires=range(num_qubits))
        for i in range(num_qubits):
            qml.RY(params[i], wires=i)
        return qml.state()

    states: List[Tensor] = []
    for state, _ in samples:
        out = circuit(state)
        states.append(out)
    return states

def state_fidelity(a: Tensor, b: Tensor) -> float:
    return float(abs((a.conj().T @ b).item()) ** 2)

def fidelity_adjacency(
    states: Sequence[Tensor],
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
#  Hybrid quantum model
# --------------------------------------------------------------------------- #
class GraphQNNQuantumModel:
    """
    Variational circuit that learns a unitary close to a target.
    """

    def __init__(self, qnn_arch: Sequence[int], target_unitary: Tensor):
        self.arch = list(qnn_arch)
        self.target = target_unitary
        self.num_qubits = qnn_arch[-1]
        self.params = torch.zeros(self.num_qubits, requires_grad=True)
        self.dev = qml.device("default.qubit", wires=self.num_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(state: Tensor) -> Tensor:
            qml.QubitStateVector(state, wires=range(self.num_qubits))
            for i in range(self.num_qubits):
                qml.RY(self.params[i], wires=i)
            return qml.state()

        self.circuit = circuit

    def forward(self, state: Tensor) -> Tensor:
        return self.circuit(state)

    def train(
        self,
        data: List[Tuple[Tensor, Tensor]],
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> None:
        optimizer = torch.optim.Adam([self.params], lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            for state, target in data:
                optimizer.zero_grad()
                pred = self.forward(state)
                loss = loss_fn(pred, target)
                loss.backward()
                optimizer.step()

    def evaluate(self, data: List[Tuple[Tensor, Tensor]]) -> float:
        loss_fn = nn.MSELoss()
        with torch.no_grad():
            losses = [loss_fn(self.forward(state), target).item() for state, target in data]
        return sum(losses) / len(losses)

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNNQuantumModel",
]
