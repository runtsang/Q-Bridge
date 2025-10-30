"""Hybrid quantum classifier that combines a random layer, data encoding, and a variational ansatz.

The build_classifier_circuit function returns a Qiskit circuit, an encoding list, parameter list,
and a set of Pauli observables, matching the interface used by the classical counterpart.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def generate_superposition_data(num_qubits: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate superposition states and binary labels derived from a sinusoidal function."""
    omega_0 = np.zeros(2 ** num_qubits, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_qubits, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_qubits), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * thetas[i]) * omega_1
    labels = (np.sin(thetas) > 0).astype(np.float32)
    return states, labels


class HybridClassifier:
    """Quantum classifier that combines an encoding, a random layer, and a variational ansatz."""

    def __init__(self, num_qubits: int, depth: int):
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)

    def evaluate(self, state_batch: np.ndarray) -> np.ndarray:
        """Simulate the circuit on a batch of classical states and return measurement expectations."""
        from qiskit import Aer, execute
        backend = Aer.get_backend("statevector_simulator")
        results = []
        for state in state_batch:
            circ = self.circuit.copy()
            # Prepare state by applying rotations based on state vector amplitude
            for i, amp in enumerate(state):
                theta = 2 * np.arccos(np.abs(amp))
                phi = np.angle(amp)
                circ.ry(theta, i)
                circ.rz(phi, i)
            job = execute(circ, backend)
            sv = job.result().get_statevector(circ)
            exp = sum(np.real(np.vdot(sv, op.to_matrix() @ sv)) for op in self.observables)
            results.append(exp)
        return np.array(results)


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a layered ansatz with a random layer and explicit encoding."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    # Random layer: apply a sequence of random single-qubit rotations
    for q in range(num_qubits):
        circuit.rx(np.random.uniform(0, 2 * np.pi), q)
        circuit.ry(np.random.uniform(0, 2 * np.pi), q)
        circuit.rz(np.random.uniform(0, 2 * np.pi), q)

    # Encoding
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            circuit.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            circuit.cx(q, q + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables
