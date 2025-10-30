"""Quantum counterpart of the hybrid fraud detection model.

The circuit is constructed with a data‑encoding stage followed by a
variational ansatz that mirrors the structure of the classical network.
The returned ``encoding`` and ``weights`` lists, together with the
``observables`` list, provide a drop‑in replacement for the classical
metadata.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[
    QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]
]:
    """Return a Qiskit circuit together with metadata.

    Parameters
    ----------
    num_qubits
        Number of qubits / input features.
    depth
        Number of variational layers.
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

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables

class FraudDetectionHybridQuantum:
    """Quantum implementation that follows the same API as the classical model."""

    def __init__(self, num_qubits: int, depth: int) -> None:
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth
        )

    def get_circuit(self) -> QuantumCircuit:
        """Return the underlying Qiskit circuit."""
        return self.circuit

    def get_encoding(self) -> List[ParameterVector]:
        """Return the encoding parameters."""
        return self.encoding

    def get_weights(self) -> List[ParameterVector]:
        """Return the variational parameters."""
        return self.weights

    def get_observables(self) -> List[SparsePauliOp]:
        """Return the measurement observables."""
        return self.observables

__all__ = [
    "build_classifier_circuit",
    "FraudDetectionHybridQuantum",
]
