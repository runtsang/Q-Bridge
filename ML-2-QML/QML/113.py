"""GraphQNN: Quantum Graph Neural Network with variational circuits.

This module implements a quantum‑based graph neural network using
PennyLane.  It exposes a variational circuit that is trained to
approximate a target unitary.  The training loop optimises the
rotation angles of each layer via a simple gradient descent
optimizer.  The module also provides utilities identical to the
classical version for generating random training data and for
constructing fidelity‑based adjacency graphs."""
from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pennylane as qml

Tensor = np.ndarray


def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Return a random unitary matrix of dimension 2**num_qubits."""
    dim = 2 ** num_qubits
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(mat)
    return q


def _random_qubit_state(num_qubits: int) -> np.ndarray:
    """Return a random pure state vector of length 2**num_qubits."""
    vec = np.random.randn(2 ** num_qubits) + 1j * np.random.randn(2 ** num_qubits)
    return vec / np.linalg.norm(vec)


def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate (input, target) pairs where target = unitary @ input."""
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    num_qubits = int(np.log2(unitary.shape[0]))
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        target = unitary @ state
        dataset.append((state, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random target unitary and training data."""
    num_qubits = qnn_arch[-1]
    target_unitary = _random_qubit_unitary(num_qubits)
    training_data = random_training_data(target_unitary, samples)
    return list(qnn_arch), None, training_data, target_unitary


def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Squared overlap between two pure states."""
    return float(abs(np.vdot(a, b)) ** 2)


def fidelity_adjacency(
    states: Sequence[np.ndarray],
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


class GraphQNN__gen123:
    """Variational graph‑based quantum neural network implemented with PennyLane."""
    def __init__(self, arch: Sequence[int], device: str = "default.qubit"):
        """
        Parameters
        ----------
        arch : Sequence[int]
            Number of qubits per layer; the last element defines the total qubits.
        device : str, default "default.qubit"
            PennyLane backend device.
        """
        self.arch = list(arch)
        self.num_qubits = self.arch[-1]
        self.wires = list(range(self.num_qubits))
        self.dev = qml.device(device, wires=self.wires)
        # Parameter layout: one (num_qubits, 3) array per layer
        self.params: List[np.ndarray] = [
            np.random.randn(self.arch[layer], 3) for layer in range(1, len(self.arch))
        ]

    def _circuit(self, inputs: np.ndarray, params: List[np.ndarray]) -> np.ndarray:
        """PennyLane quantum circuit returning the final state vector."""
        @qml.qnode(self.dev)
        def circuit():
            qml.StatePrep(inputs, wires=self.wires)
            for layer_idx, layer_params in enumerate(params):
                for qubit, angles in enumerate(layer_params):
                    qml.Rot(*angles, wires=layer_idx * self.num_qubits + qubit)
                # Simple entanglement pattern: nearest‑neighbour CNOTs
                for q in range(self.num_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            return qml.state()
        return circuit()

    def feedforward(self, inputs: np.ndarray) -> np.ndarray:
        """Run the current variational circuit on the supplied state."""
        return self._circuit(inputs, self.params)

    def train_qnn(
        self,
        target_unitary: np.ndarray,
        training_data: List[Tuple[np.ndarray, np.ndarray]],
        epochs: int = 200,
        lr: float = 0.01,
    ) -> List[np.ndarray]:
        """Optimize the circuit parameters to approximate `target_unitary`."""
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        for _ in range(epochs):
            for state, target in training_data:
                def loss_fn(params):
                    out = self._circuit(state, params)
                    fid = abs(np.vdot(out, target)) ** 2
                    return 1.0 - fid
                self.params = opt.step(loss_fn, self.params)
        return self.params

    def state_fidelity(self, a: np.ndarray, b: np.ndarray) -> float:
        return state_fidelity(a, b)

    def fidelity_adjacency(
        self,
        states: Sequence[np.ndarray],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        return random_network(qnn_arch, samples)

    @staticmethod
    def random_training_data(unitary: np.ndarray, samples: int):
        return random_training_data(unitary, samples)

    @staticmethod
    def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
        return state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[np.ndarray],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)


__all__ = [
    "GraphQNN__gen123",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
