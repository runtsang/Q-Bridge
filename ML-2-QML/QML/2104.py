"""Quantum implementation of GraphQNN using Pennylane.

This module mirrors the classical `GraphQNN__gen318` but replaces the
feed‑forward computation with a variational quantum circuit.
The architecture is specified by a list of integers; the final
integer determines the number of qubits in the output register.
The class provides methods for random network generation,
forward propagation, state fidelity, and fidelity‑based graph
construction.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Any

import networkx as nx
import pennylane as qml
import pennylane.numpy as np

# --------------------------------------------------------------------------- #
# 1. Quantum utilities
# --------------------------------------------------------------------------- #

def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate `(state, target)` pairs where target = U @ state."""
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    dim = unitary.shape[0]
    for _ in range(samples):
        state = np.random.normal(size=(dim, 1))
        state /= np.linalg.norm(state)
        target = unitary @ state
        dataset.append((state, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Return architecture, list of parameter matrices, training data, and target unitary."""
    num_qubits = qnn_arch[-1]
    dim = 2 ** num_qubits
    random_matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    q, _ = np.linalg.qr(random_matrix)  # Q is a unitary matrix
    unitary = q
    training_data = random_training_data(unitary, samples)

    params: List[np.ndarray] = [np.random.normal(size=(num_qubits, num_qubits)) for _ in range(1, len(qnn_arch))]
    return list(qnn_arch), params, training_data, unitary


def _layer_circuit(num_qubits: int, params: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Apply a single variational layer to the input state."""
    @qml.qnode(qml.device("default.qubit", wires=num_qubits), interface="autograd")
    def circuit():
        qml.QubitStateVector(state, wires=range(num_qubits))
        # Single‑qubit rotations
        for q in range(num_qubits):
            qml.Rot(params[q, 0], params[q, 1], params[q, 2], wires=q)
        # Entanglement
        for q in range(num_qubits - 1):
            qml.CNOT(wires=[q, q + 1])
        return qml.state()
    return circuit()


def feedforward(
    qnn_arch: Sequence[int],
    params_list: Sequence[np.ndarray],
    samples: Iterable[Tuple[np.ndarray, np.ndarray]],
) -> List[List[np.ndarray]]:
    """Return the state after each layer for each sample."""
    num_qubits = qnn_arch[-1]
    stored_states: List[List[np.ndarray]] = []
    for state, _ in samples:
        layer_states: List[np.ndarray] = [state]
        current = state
        for params in params_list:
            current = _layer_circuit(num_qubits, params, current)
            layer_states.append(current)
        stored_states.append(layer_states)
    return stored_states


def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Return squared overlap between two quantum states."""
    return float(np.abs(np.vdot(a, b)) ** 2)


def fidelity_adjacency(
    states: Sequence[np.ndarray],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted graph from state fidelities."""
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
# 2. Quantum GraphQNN class
# --------------------------------------------------------------------------- #

class GraphQNN__gen318:
    """Quantum variant of the GraphQNN architecture.

    Parameters
    ----------
    arch : Sequence[int]
        Architecture list, e.g. ``[n_inputs,..., n_outputs]``.  The
        number of output qubits equals ``arch[-1]``.
    """

    def __init__(self, arch: Sequence[int], device: Any | None = None):
        self.arch = list(arch)
        self.num_qubits = self.arch[-1]
        self.device = device or qml.device("default.qubit", wires=self.num_qubits)
        # Parameters for each layer: a matrix of shape (num_qubits, num_qubits)
        self.params: List[np.ndarray] = [
            np.random.normal(size=(self.num_qubits, self.num_qubits))
            for _ in range(1, len(self.arch))
        ]

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Return the final quantum state for a single input."""
        @qml.qnode(self.device, interface="autograd")
        def circuit():
            qml.QubitStateVector(state, wires=range(self.num_qubits))
            for params in self.params:
                for q in range(self.num_qubits):
                    qml.Rot(params[q, 0], params[q, 1], params[q, 2], wires=q)
                for q in range(self.num_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            return qml.state()
        return circuit()

    def predict(self, samples: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List[List[np.ndarray]]:
        """Return the state after each layer for each sample."""
        stored_states: List[List[np.ndarray]] = []
        for state, _ in samples:
            layer_states: List[np.ndarray] = [state]
            current = state
            for params in self.params:
                @qml.qnode(self.device, interface="autograd")
                def layer_circuit():
                    qml.QubitStateVector(current, wires=range(self.num_qubits))
                    for q in range(self.num_qubits):
                        qml.Rot(params[q, 0], params[q, 1], params[q, 2], wires=q)
                    for q in range(self.num_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])
                    return qml.state()
                current = layer_circuit()
                layer_states.append(current)
            stored_states.append(layer_states)
        return stored_states

    def fidelity_graph(
        self,
        states: Sequence[np.ndarray],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Return the fidelity‑based adjacency graph of the given states."""
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def train_random(self, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate a random training set based on a random target unitary."""
        _, _, dataset, _ = random_network(self.arch, samples)
        return dataset


__all__ = [
    "GraphQNN__gen318",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
