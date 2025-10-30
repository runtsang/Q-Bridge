"""GraphQNN with a parameterized quantum circuit.

The quantum side uses PennyLane to build a variational circuit that
is trained to approximate a target unitary.  The module includes
data generation, fidelity utilities, and a simple training routine.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml

def _random_unitary(num_qubits: int) -> np.ndarray:
    """Generate a random unitary matrix via QR decomposition."""
    dim = 2 ** num_qubits
    random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, r = np.linalg.qr(random_matrix)
    d = np.diag(r)
    ph = d / np.abs(d)
    return q @ np.diag(ph)


def _random_qubit_state(num_qubits: int) -> np.ndarray:
    """Sample a random pure state vector."""
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return vec


def random_training_data(
    unitary: np.ndarray,
    samples: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate input–output state pairs using the target unitary."""
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(samples):
        state = _random_qubit_state(len(unitary) // 2)
        target = unitary @ state
        dataset.append((state, target))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    """Instantiate a variational circuit and a training set."""
    num_qubits = qnn_arch[-1]
    target_unitary = _random_unitary(num_qubits)
    training_data = random_training_data(target_unitary, samples)

    # Initialise one set of rotation parameters per qubit
    params: List[np.ndarray] = []
    for n in qnn_arch:
        # 3 angles per qubit for a generic SU(2) rotation
        params.append(np.random.uniform(0, 2 * np.pi, size=(n, 3)))

    return qnn_arch, params, training_data, target_unitary


def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the squared overlap between two pure state vectors."""
    return float(np.abs(np.vdot(a, b)) ** 2)


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


def feedforward(
    qnn_arch: Sequence[int],
    params: Sequence[np.ndarray],
    samples: Iterable[Tuple[np.ndarray, np.ndarray]],
) -> List[np.ndarray]:
    """Run the circuit on each input and return the final output states."""
    dev = qml.device("default.qubit", wires=qnn_arch[-1])

    @qml.qnode(dev)
    def circuit(state, params):
        qml.StatePrep(state, wires=range(qnn_arch[-1]))
        for i in range(qnn_arch[-1]):
            theta, phi, lam = params[i]
            qml.Rot(theta, phi, lam, wires=i)
        return qml.state()

    outputs: List[np.ndarray] = []
    for sample, _ in samples:
        outputs.append(circuit(sample, params))
    return outputs


class GraphQNNQuantumModel:
    """A simple variational quantum circuit built with PennyLane."""

    def __init__(self, arch: Sequence[int], device: qml.Device | None = None):
        self.arch = list(arch)
        self.num_qubits = arch[-1]
        self.dev = device or qml.device("default.qubit", wires=self.num_qubits)

        # One set of three angles per qubit
        self.params = np.random.uniform(0, 2 * np.pi, size=(self.num_qubits, 3))

        self.qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, state: np.ndarray, params: np.ndarray):
        """Variational circuit that applies an SU(2) rotation to each qubit."""
        qml.StatePrep(state, wires=range(self.num_qubits))
        for i in range(self.num_qubits):
            theta, phi, lam = params[i]
            qml.Rot(theta, phi, lam, wires=i)
        return qml.state()

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Return the circuit’s output state vector for the given input."""
        return self.qnode(state, self.params)

    def train(
        self,
        training_data: List[Tuple[np.ndarray, np.ndarray]],
        lr: float = 0.01,
        epochs: int = 200,
        verbose: bool = False,
    ) -> List[float]:
        """Optimise the rotation angles to reproduce the target unitary."""
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        loss_history: List[float] = []

        for epoch in range(epochs):
            for inp, target in training_data:
                def loss_fn(params):
                    pred = self.qnode(inp, params)
                    return np.mean((pred - target) ** 2)

                self.params = opt.step(loss_fn, self.params)

            epoch_loss = np.mean(
                [np.mean((self.qnode(inp, self.params) - target) ** 2)
                 for inp, target in training_data]
            )
            loss_history.append(epoch_loss)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs} – loss: {epoch_loss:.6f}")

        return loss_history


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNNQuantumModel",
]
