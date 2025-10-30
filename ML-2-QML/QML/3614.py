"""Quantum graph neural network with variational layers.

This module builds a hybrid architecture that mirrors the classical
GraphQNNHybrid but replaces the linear layers with a parametrised
variational circuit.  It uses Pennylane for the quantum backend
and includes support for regression on the superposition dataset
from the quantum regression example.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
State = torch.Tensor

# --------------------------------------------------------------------------- #
# 1. Random unitary / state helpers
# --------------------------------------------------------------------------- #
def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Return a random unitary matrix of size 2**num_qubits."""
    dim = 2 ** num_qubits
    return qml.random_unitary(dim)


def _random_qubit_state(num_qubits: int) -> np.ndarray:
    """Return a random pure state of size 2**num_qubits."""
    dim = 2 ** num_qubits
    state = np.random.normal(size=dim) + 1j * np.random.normal(size=dim)
    state /= np.linalg.norm(state)
    return state


def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate training pairs by applying ``unitary`` to random states."""
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(samples):
        state = _random_qubit_state(unitary.shape[0].bit_length() - 1)
        dataset.append((state, unitary @ state))
    return dataset


def random_network(arch: Sequence[int], samples: int) -> Tuple[Sequence[int], List[List[np.ndarray]], List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Create a random network of unitaries and a training dataset for the final layer."""
    target_unitary = _random_qubit_unitary(arch[-1])
    dataset = random_training_data(target_unitary, samples)
    unitaries: List[List[np.ndarray]] = [[]]
    for layer in range(1, len(arch)):
        num_inputs = arch[layer - 1]
        num_outputs = arch[layer]
        layer_ops: List[np.ndarray] = []
        for _ in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            layer_ops.append(op)
        unitaries.append(layer_ops)
    return arch, unitaries, dataset, target_unitary


# --------------------------------------------------------------------------- #
# 2. Forward propagation helpers
# --------------------------------------------------------------------------- #
def feedforward(
    arch: Sequence[int],
    unitaries: Sequence[Sequence[np.ndarray]],
    samples: Iterable[Tuple[np.ndarray, np.ndarray]],
) -> List[List[np.ndarray]]:
    """Propagate each training state through the network of unitaries."""
    stored_states: List[List[np.ndarray]] = []
    for state, _ in samples:
        layerwise = [state]
        current = state
        for layer in range(1, len(arch)):
            ops = unitaries[layer]
            for op in ops:
                current = op @ current
            layerwise.append(current)
        stored_states.append(layerwise)
    return stored_states


# --------------------------------------------------------------------------- #
# 3. Fidelity utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the absolute squared overlap between two pure states."""
    return abs(np.vdot(a, b)) ** 2


def fidelity_adjacency(
    states: Sequence[np.ndarray],
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


# --------------------------------------------------------------------------- #
# 4. Quantum‐aware hybrid network
# --------------------------------------------------------------------------- #
class GraphQNNHybrid(nn.Module):
    """A hybrid graph neural network that uses a Pennylane variational
    circuit as the core encoder.  The network accepts a batch of classical
    feature vectors and maps them to quantum states via rotation gates.
    """

    def __init__(self, arch: Sequence[int], dev: str = "default.qubit"):
        super().__init__()
        self.arch = list(arch)
        self.n_wires = self.arch[-1]
        self.dev = qml.device(dev, wires=self.n_wires)
        self._circuit = self._build_circuit()
        self.head = nn.Linear(self.n_wires, 1)

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(x: Tensor):
            # Encode each feature as an RX rotation
            for i in range(self.n_wires):
                qml.RX(x[i], wires=i)
            # Strongly entangling layer (no trainable weights for simplicity)
            qml.StronglyEntanglingLayers(weights=None, wires=range(self.n_wires))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]
        return circuit

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: ``x`` has shape (batch, n_wires)."""
        batch_out = self._circuit(x)  # Pennylane handles batching
        return self.head(batch_out).squeeze(-1)


# --------------------------------------------------------------------------- #
# 5. Regression dataset (quantum)
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0…0> + e^{i phi} sin(theta)|1…1>."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that returns a complex state tensor and a scalar target."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


__all__ = [
    "GraphQNNHybrid",
    "generate_superposition_data",
    "RegressionDataset",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
