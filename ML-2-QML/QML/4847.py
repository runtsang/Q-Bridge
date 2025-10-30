"""Hybrid graph neural network utilities – quantum implementation."""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple, Callable

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Pauli


def _random_unitary(num_qubits: int) -> np.ndarray:
    """Generate a random unitary matrix using QR decomposition."""
    dim = 2 ** num_qubits
    matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(matrix)
    return q


def random_quantum_network(qnn_arch: Sequence[int], samples: int):
    """
    Build a list of parameterized QuantumCircuit objects representing a
    variational graph neural network. The last circuit is treated as the
    target unitary for training data generation.
    """
    circuits: List[QuantumCircuit] = []
    target_unitary = _random_unitary(qnn_arch[-1])

    # Build intermediate layers
    for n in qnn_arch[:-1]:
        qc = QuantumCircuit(n)
        for qubit in range(n):
            theta = Parameter(f"theta_{qubit}")
            qc.ry(theta, qubit)
            qc.rx(theta, qubit)
        # Add a simple entangling pattern
        for i in range(n - 1):
            qc.cx(i, i + 1)
        circuits.append(qc)

    # Append the target unitary as a fixed circuit
    target_qc = QuantumCircuit(qnn_arch[-1])
    target_qc.unitary(target_unitary, list(range(qnn_arch[-1])))
    circuits.append(target_qc)

    training_data = random_training_data(target_unitary, samples)
    return list(qnn_arch), circuits, training_data, target_unitary


def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[Statevector, Statevector]]:
    """Generate random input states and their transformations under a unitary."""
    dataset: List[Tuple[Statevector, Statevector]] = []
    dim = unitary.shape[0]
    for _ in range(samples):
        vec = np.random.randn(dim) + 1j * np.random.randn(dim)
        vec /= np.linalg.norm(vec)
        input_sv = Statevector(vec)
        output_sv = input_sv.evolve(unitary)
        dataset.append((input_sv, output_sv))
    return dataset


def _layer_apply(circuit: QuantumCircuit, state: Statevector) -> Statevector:
    """Apply a single layer circuit to a statevector."""
    backend = Aer.get_backend("statevector_simulator")
    job = execute(circuit, backend, initial_state=state.data)
    return Statevector(job.result().get_statevector())


def feedforward_quantum(qnn_arch: Sequence[int], circuits: Sequence[QuantumCircuit], samples: Iterable[Tuple[Statevector, Statevector]]) -> List[List[Statevector]]:
    """Propagate a state through each layer of the quantum network."""
    state_seqs: List[List[Statevector]] = []
    for input_state, _ in samples:
        layer_states: List[Statevector] = [input_state]
        current = input_state
        for circuit in circuits[:-1]:  # skip target layer
            current = _layer_apply(circuit, current)
            layer_states.append(current)
        state_seqs.append(layer_states)
    return state_seqs


def state_fidelity_quantum(a: Statevector, b: Statevector) -> float:
    """Squared overlap between two pure statevectors."""
    return abs(np.vdot(a.data, b.data)) ** 2


def fidelity_adjacency_quantum(states: Sequence[Statevector], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph from pairwise quantum state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity_quantum(s_i, s_j)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G


class FastBaseEstimatorQuantum:
    """Evaluate expectation values of Pauli operators on a parameterized circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._params):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self._params, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[Pauli], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        results: List[List[complex]] = []
        for vals in parameter_sets:
            bound = self._bind(vals)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class EstimatorQNNQuantum:
    """Simple quantum estimator using a single‑qubit circuit with Ry and Rx."""
    def __init__(self) -> None:
        self.params = [Parameter("theta_input"), Parameter("theta_weight")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(self.params[0], 0)
        qc.rx(self.params[1], 0)
        self.circuit = qc
        self.observable = Pauli("Y")

    def evaluate(self, param_set: Sequence[float]) -> complex:
        bound = self.circuit.assign_parameters(dict(zip(self.params, param_set)), inplace=False)
        state = Statevector.from_instruction(bound)
        return state.expectation_value(self.observable)


__all__ = [
    "GraphQNNHybrid",
    "random_quantum_network",
    "feedforward_quantum",
    "state_fidelity_quantum",
    "fidelity_adjacency_quantum",
    "FastBaseEstimatorQuantum",
    "EstimatorQNNQuantum",
]
