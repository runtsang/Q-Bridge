"""Quantum graph neural network implementation based on PennyLane.

The :class:`GraphQNN` class builds a variational circuit that mimics
the structure of a classical feed‑forward network.  Each layer
consists of a block of parameterised single‑qubit rotations followed
by a layer‑dependent entanglement pattern.  The class provides
training utilities that minimise a fidelity loss between the circuit
output and a target unitary, and graph‑based analysis tools that
mirror the classical version."""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import pennylane as qml
import numpy as np
import networkx as nx

# -------------------------------------------------------------- #
#  Helper functions for random unitary generation
# -------------------------------------------------------------- #
def _random_unitary(dim: int) -> np.ndarray:
    """Generate a Haar‑random unitary matrix."""
    x = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim))
    q, _ = np.linalg.qr(x)
    return q


def _random_angles(n_params: int) -> np.ndarray:
    """Random rotation angles in [0, 2π)."""
    return np.random.uniform(0, 2 * np.pi, size=n_params)


# -------------------------------------------------------------- #
#  Quantum Graph Neural Network
# -------------------------------------------------------------- #
class GraphQNN:
    """A variational circuit that maps a classical GNN structure
    onto a quantum circuit.  The network is defined by its
    architecture (number of qubits per layer) and a set of
    parameterised rotation gates per qubit.
    """

    def __init__(self, architecture: Sequence[int], device: str = "default.qubit") -> None:
        self.architecture = list(architecture)
        self.max_wires = max(self.architecture)
        self.dev = qml.device(device, wires=self.max_wires)
        self.params = self._init_params()
        self.circuit = self._build_circuit()

    # ------------------------------------------------------------------ #
    #  Parameter initialisation
    # ------------------------------------------------------------------ #
    def _init_params(self) -> np.ndarray:
        """Initialise rotation angles for each qubit in each layer."""
        params = []
        for n_qubits in self.architecture:
            params.append(_random_angles(n_qubits * 3))  # RX,RZ,RY per qubit
        return np.array(params)

    # ------------------------------------------------------------------ #
    #  Quantum circuit construction
    # ------------------------------------------------------------------ #
    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(input_state: np.ndarray, params: np.ndarray):
            # Load input state
            qml.QubitStateVector(input_state, wires=range(len(input_state)))
            # Layered architecture
            for l, n_qubits in enumerate(self.architecture):
                # Rotation block
                for q in range(n_qubits):
                    idx = q * 3
                    qml.RX(params[l][idx], wires=q)
                    qml.RZ(params[l][idx + 1], wires=q)
                    qml.RY(params[l][idx + 2], wires=q)
                # Entanglement pattern
                if n_qubits > 1:
                    for q in range(n_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])
            # Return state vector
            return qml.state()
        return circuit

    # ------------------------------------------------------------------ #
    #  Static helpers for data generation
    # ------------------------------------------------------------------ #
    @staticmethod
    def random_training_data(target_unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate training pairs (input, target) for a given unitary."""
        dim = target_unitary.shape[0]
        dataset: List[Tuple[np.ndarray, np.ndarray]] = []
        for _ in range(samples):
            state = np.random.randn(dim) + 1j * np.random.randn(dim)
            state /= np.linalg.norm(state)
            target = target_unitary @ state
            dataset.append((state, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Return architecture, initial parameters, training data, and target unitary."""
        target_unitary = _random_unitary(2 ** max(qnn_arch))
        training_data = GraphQNN.random_training_data(target_unitary, samples)
        return list(qnn_arch), target_unitary, training_data, target_unitary

    # ------------------------------------------------------------------ #
    #  Forward propagation
    # ------------------------------------------------------------------ #
    def feedforward(self, samples: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List[List[np.ndarray]]:
        """Return state vectors for every layer for each input sample."""
        stored: List[List[np.ndarray]] = []
        for state, _ in samples:
            layerwise = [state]
            current = state
            # Iterate through layers
            for l, n_qubits in enumerate(self.architecture):
                # Apply rotations
                for q in range(n_qubits):
                    idx = q * 3
                    current = qml.RX(self.params[l][idx], wires=q).data
                    current = qml.RZ(self.params[l][idx + 1], wires=q).data
                    current = qml.RY(self.params[l][idx + 2], wires=q).data
                # Entanglement
                if n_qubits > 1:
                    for q in range(n_qubits - 1):
                        current = qml.CNOT(wires=[q, q + 1]).data
                layerwise.append(current)
            stored.append(layerwise)
        return stored

    # ------------------------------------------------------------------ #
    #  Fidelity utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
        """Return the squared absolute overlap between two pure states."""
        return float(abs(np.vdot(a, b)) ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[np.ndarray],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------ #
    #  Training utilities
    # ------------------------------------------------------------------ #
    def train(
        self,
        training_data: Iterable[Tuple[np.ndarray, np.ndarray]],
        epochs: int = 200,
        lr: float = 1e-2,
    ) -> None:
        """Gradient‑based optimisation of rotation angles to match the target unitary."""
        opt = qml.AdamOptimizer(stepsize=lr)
        for _ in range(epochs):
            for state, target in training_data:
                def loss_fn(p):
                    pred = self.circuit(state, p)
                    return 1 - GraphQNN.state_fidelity(pred, target)
                self.params = opt.step(loss_fn, self.params)

    def evaluate(self, samples: Iterable[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Return average fidelity loss over the samples."""
        losses = []
        for state, target in samples:
            pred = self.circuit(state, self.params)
            losses.append(1 - GraphQNN.state_fidelity(pred, target))
        return np.mean(losses)

    # ------------------------------------------------------------------ #
    #  Utility: convert to Qiskit circuit
    # ------------------------------------------------------------------ #
    def to_qiskit_circuit(self):
        """Export the current variational circuit as a Qiskit QuantumCircuit."""
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(self.max_wires)
        for l, n_qubits in enumerate(self.architecture):
            for q in range(n_qubits):
                idx = q * 3
                qc.rx(self.params[l][idx], q)
                qc.rz(self.params[l][idx + 1], q)
                qc.ry(self.params[l][idx + 2], q)
            if n_qubits > 1:
                for q in range(n_qubits - 1):
                    qc.cx(q, q + 1)
        return qc


__all__ = ["GraphQNN"]
