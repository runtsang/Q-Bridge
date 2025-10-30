"""Quantum self‑attention classifier built with Qiskit."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, Tuple, List


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a variational circuit that emulates a self‑attention block.

    Returns:
        circuit: QuantumCircuit implementing rotation, entanglement, and depth‑wise layers.
        encoding: list of ParameterVector objects for data encoding.
        weights: list of ParameterVector objects for variational parameters.
        observables: list of Pauli‑Z observables for classification.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


class QuantumSelfAttentionClassifier:
    """Variational classifier that mirrors the classical attention block."""

    def __init__(self, n_qubits: int, depth: int):
        self.n_qubits = n_qubits
        self.depth = depth
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        # Rotation layer
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # Entanglement layer
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        # Depth‑wise variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.n_qubits):
                circuit.ry(entangle_params[idx], qubit)
                idx += 1
            for qubit in range(self.n_qubits - 1):
                circuit.cz(qubit, qubit + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)


# Default backend and instance
backend = qiskit.Aer.get_backend("qasm_simulator")
attention_classifier = QuantumSelfAttentionClassifier(n_qubits=4, depth=2)

__all__ = ["build_classifier_circuit", "QuantumSelfAttentionClassifier"]
