"""
GraphQuantumNeuralNetwork: quantum version with a variational circuit
and a fidelity‑based graph construction.

This implementation builds a parameterised circuit with a configurable
depth of Trotter‑ised rotation layers.  The class can generate a random
target unitary, run a batch of input states, compute a fidelity‑based
adjacency graph, and optimise the circuit parameters to approximate the
target using a fidelity loss.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def _random_qubit_unitary(num_qubits: int) -> qml.QubitUnitary:
    """Return a random unitary as a PennyLane QubitUnitary."""
    dim = 2 ** num_qubits
    matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    matrix, _ = np.linalg.qr(matrix)
    return qml.QubitUnitary(matrix, wires=range(num_qubits))


def random_training_data(unitary: qml.QubitUnitary, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate (state, unitary*state) pairs for training."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    num_qubits = len(unitary.wires)
    for _ in range(samples):
        state = np.random.randn(2 ** num_qubits) + 1j * np.random.randn(2 ** num_qubits)
        state /= np.linalg.norm(state)
        target = unitary.matrix @ state
        dataset.append((torch.from_numpy(state).float(), torch.from_numpy(target).float()))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int) -> tuple[list[int], List[Tuple[Tensor, Tensor]], qml.QubitUnitary]:
    """Create a random target unitary and a training set."""
    num_qubits = qnn_arch[-1]
    target_unitary = _random_qubit_unitary(num_qubits)
    training_data = random_training_data(target_unitary, samples)
    return list(qnn_arch), training_data, target_unitary


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
        fid = abs((state_i @ state_j.conj()).item()) ** 2
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class GraphQuantumNeuralNetwork(nn.Module):
    """Parameterized quantum neural network with a variational circuit."""

    def __init__(
        self,
        arch: Sequence[int],
        depth: int = 2,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.arch = list(arch)
        self.depth = depth
        self.device = torch.device(device)

        self.num_qubits = self.arch[-1]
        self.wires = list(range(self.num_qubits))

        # Parameters for rotation angles: shape (depth, num_qubits, 3)
        self.theta = nn.Parameter(
            torch.randn(depth, self.num_qubits, 3, dtype=torch.float32)
        )

        # Define the quantum device
        self.dev = qml.device("default.qubit", wires=self.num_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: Tensor, theta: Tensor) -> Tensor:
            """Variational circuit with depth repetitions."""
            # Encode the input state via a simple rotation
            for i in range(self.num_qubits):
                qml.RZ(inputs[i], wires=i)

            # Repeated rotation layers
            for d in range(self.depth):
                for i in range(self.num_qubits):
                    qml.RX(theta[d, i, 0], wires=i)
                    qml.RY(theta[d, i, 1], wires=i)
                    qml.RZ(theta[d, i, 2], wires=i)
                # Entangling layer
                for i in range(self.num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.num_qubits - 1, 0])

            return qml.state()

        self.circuit = circuit

    @staticmethod
    def random_network(arch: Sequence[int], samples: int) -> tuple[list[int], List[Tuple[Tensor, Tensor]], qml.QubitUnitary]:
        """Proxy to the module‑level helper."""
        return random_network(arch, samples)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def forward(self, inputs: Tensor) -> Tensor:
        """Run the variational circuit on a batch of input states."""
        # Ensure inputs are of shape (batch, num_qubits)
        batch_size = inputs.shape[0]
        outputs = []
        for i in range(batch_size):
            state = self.circuit(inputs[i], self.theta)
            outputs.append(state)
        return torch.stack(outputs)

    def hybrid_loss(
        self,
        outputs: Tensor,
        targets: Tensor,
        target_unitary: qml.QubitUnitary,
    ) -> Tensor:
        """Hybrid loss: MSE + (1 - fidelity) between circuit output and target unitary."""
        # MSE on the state amplitudes
        mse = F.mse_loss(outputs, targets)

        # Fidelity between each output state and the target unitary applied to the same input
        fidelities = []
        for out, tgt in zip(outputs, targets):
            # target state = U * input
            fid = abs((out @ tgt.conj()).item()) ** 2
            fidelities.append(fid)
        fidelity = torch.mean(torch.tensor(fidelities, device=self.device))
        return mse + (1.0 - fidelity)

    def train_step(
        self,
        optimizer: torch.optim.Optimizer,
        batch: List[Tuple[Tensor, Tensor]],
        target_unitary: qml.QubitUnitary,
    ) -> Tensor:
        """Single optimizer step on a batch with hybrid loss."""
        optimizer.zero_grad()
        inputs = torch.stack([b[0] for b in batch]).to(self.device)
        targets = torch.stack([b[1] for b in batch]).to(self.device)
        outputs = self.forward(inputs)
        loss = self.hybrid_loss(outputs, targets, target_unitary)
        loss.backward()
        optimizer.step()
        return loss


__all__ = [
    "GraphQuantumNeuralNetwork",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
]
