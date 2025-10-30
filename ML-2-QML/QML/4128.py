"""Hybrid quantum classifier mirroring the classical interface, with data‑uploading and variational layers."""

from __future__ import annotations

from typing import List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """Construct a hybrid ansatz: data encoding + random + controlled rotations."""
    # Encoding
    encoding = ParameterVector("x", num_qubits)
    # Variational parameters
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    # Data‑uploading layer
    for qubit, param in zip(range(num_qubits), encoding):
        circuit.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        # Random rotation layer (simulated by random angles)
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        # Entangling block
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables (Z on each qubit)
    observables = [SparsePauliOp(f"I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


class HybridQuantumClassifier:
    """Quantum counterpart that exposes the same metadata as HybridClassifier."""

    def __init__(self, num_qubits: int = 4, depth: int = 2) -> None:
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)

    def get_circuit(self) -> QuantumCircuit:
        return self.circuit

    def get_metadata(self) -> Tuple[List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        return self.encoding, self.weights, self.observables


__all__ = ["build_classifier_circuit", "HybridQuantumClassifier"]
