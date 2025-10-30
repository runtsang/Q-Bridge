"""QuantumHybridClassifier – quantum circuit implementation.

The module defines a class `QuantumHybridClassifier` that builds a
variational quantum circuit.  It mirrors the API of the classical
`build_classifier_circuit` but returns a Qiskit `QuantumCircuit`
along with encoding, weight parameters, and measurement
observables.  The circuit implements a data‑uploading ansatz followed
by a depth‑controlled variational block with CZ entanglement.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class QuantumHybridClassifier:
    """Container that builds a quantum variational circuit."""
    def __init__(self, num_qubits: int, depth: int) -> None:
        self.num_qubits = num_qubits
        self.depth = depth

    def build(self) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        return build_classifier_circuit(self.num_qubits, self.depth)

def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables

__all__ = ["QuantumHybridClassifier", "build_classifier_circuit"]
