"""Quantum implementation of the fraud detection pipeline.

The `FraudDetectionHybrid` class builds a parameterised quantum circuit that
encodes the input data and applies a depth‑controlled variational ansatz.
The returned `QuantumCircuit` can be executed on any Qiskit backend or
simulated with statevector or qasm simulators.

This module intentionally mirrors the classical interface so that the
hybrid framework can switch seamlessly between quantum and classical
backends.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a simple layered ansatz with explicit encoding and variational parameters."""
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


class FraudDetectionHybrid:
    """
    Quantum fraud detection circuit mirroring the classical interface.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features) to encode.
    depth : int
        Depth of the variational ansatz.
    """

    def __init__(self, num_qubits: int, depth: int) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding, self.weights, self.observables = (
            build_classifier_circuit(num_qubits, depth)
        )

    def get_circuit(self) -> QuantumCircuit:
        """Return the parameterised circuit for execution or simulation."""
        return self.circuit


__all__ = ["build_classifier_circuit", "FraudDetectionHybrid"]
