"""Quantum graph neural network with estimator support.

This module implements a quantum version of the GraphQNNGen112
interface.  It builds layered parameterised unitaries and provides
Qiskit EstimatorQNN integration.  Random network generation,
training data synthesis, and fidelity‑based graph construction are
mirrored from the classical side to keep a unified API.
"""

from __future__ import annotations

import itertools
import numpy as np
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import qiskit as qk
import qiskit.quantum_info as qi
import qiskit_machine_learning.neural_networks as qml_nn
from qiskit.primitives import Estimator as StatevectorEstimator

State = qi.Statevector
Tensor = np.ndarray


def _random_qubit_unitary(num_qubits: int) -> State:
    dim = 2**num_qubits
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    mat, _ = np.linalg.qr(mat)  # QR gives a random unitary
    return State.from_label("0"*num_qubits).compose(mat)


def _random_qubit_state(num_qubits: int) -> State:
    dim = 2**num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return State(vec)


class GraphQNNGen112:
    """Quantum implementation of a graph‑neural‑network style network.

    Parameters
    ----------
    arch : Sequence[int]
        Number of qubits per layer.  The first element is the input
        register size; each subsequent element specifies the number
        of output qubits for that layer.
    """

    def __init__(self, arch: Sequence[int]) -> None:
        self.arch = list(arch)
        self.circuits: List[qk.QuantumCircuit] = self._build_circuits()

    def _build_circuits(self) -> List[qk.QuantumCircuit]:
        """Create a list of parameterised circuits, one per layer."""
        circuits: List[qk.QuantumCircuit] = []
        for layer in range(1, len(self.arch)):
            n_in = self.arch[layer - 1]
            n_out = self.arch[layer]
            qc = qk.QuantumCircuit(n_in + n_out)
            # Randomly initialise a unitary on the whole register
            unitary = _random_qubit_unitary(n_in + n_out)
            # Apply it via a unitary gate
            qc.unitary(unitary.data, qk.QuantumRegister(n_in + n_out))
            circuits.append(qc)
        return circuits

    def forward(self, state: State) -> List[State]:
        """Propagate a state through all layers, returning each intermediate."""
        states: List[State] = [state]
        current = state
        for qc in self.circuits:
            # Append zero‑qubit register for outputs
            zero_reg = _random_qubit_state(len(qc.qubits) - len(state.data.shape))
            # Combine state with zeros
            combined = current.compose(zero_reg, qubits=range(len(current.data.shape)))
            # Apply the circuit
            combined = combined.compose(qc, qubits=range(len(combined.data.shape)))
            # Partial trace out the input qubits
            keep = list(range(len(state.data.shape)))
            current = combined.ptrace(keep)
            states.append(current)
        return states

    @staticmethod
    def state_fidelity(a: State, b: State) -> float:
        """Return the absolute squared overlap between pure states."""
        return abs((a.dag() @ b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[State],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Create a weighted adjacency graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(
            enumerate(states), 2
        ):
            fid = GraphQNNGen112.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def random_training_data(
        unitary: State, samples: int
    ) -> List[Tuple[State, State]]:
        """Generate synthetic training data for a target unitary."""
        dataset: List[Tuple[State, State]] = []
        for _ in range(samples):
            st = _random_qubit_state(len(unitary.data.shape))
            dataset.append((st, unitary @ st))
        return dataset

    @staticmethod
    def random_network(arch: Sequence[int], samples: int):
        """Generate a random quantum network and training data."""
        target_unitary = _random_qubit_unitary(arch[-1])
        training_data = GraphQNNGen112.random_training_data(target_unitary, samples)

        circuits: List[qk.QuantumCircuit] = []
        for layer in range(1, len(arch)):
            n_in = arch[layer - 1]
            n_out = arch[layer]
            qc = qk.QuantumCircuit(n_in + n_out)
            unitary = _random_qubit_unitary(n_in + n_out)
            qc.unitary(unitary.data, qk.QuantumRegister(n_in + n_out))
            circuits.append(qc)

        return arch, circuits, training_data, target_unitary


def EstimatorQNN() -> qml_nn.EstimatorQNN:
    """Return a Qiskit EstimatorQNN instance built from a random network.

    The circuit is constructed with a single 1‑qubit layer and a
    Pauli‑Y observable, mirroring the original EstimatorQNN example.
    """
    # Simple 1‑qubit circuit for demonstration
    qc = qk.QuantumCircuit(1)
    qc.h(0)
    param = qk.circuit.Parameter("theta")
    qc.ry(param, 0)

    observable = qi.SparsePauliOp.from_list([("Y", 1)])

    estimator = StatevectorEstimator()
    return qml_nn.EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[param],
        weight_params=[param],
        estimator=estimator,
    )


__all__ = [
    "GraphQNNGen112",
    "EstimatorQNN",
]
