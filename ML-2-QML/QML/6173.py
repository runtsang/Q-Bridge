"""Quantum circuit factory for the incremental data‑uploading classifier."""

from __future__ import annotations

from typing import Iterable, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

__all__ = ["QuantumClassifierModel"]

class QuantumClassifierModel:
    """
    Builds a variational quantum circuit with data re‑uploading for binary classification.
    """

    def __init__(self, num_qubits: int, depth: int):
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding_params, self.weight_params, self.observables = self._build_circuit()

    def _build_circuit(self) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        circuit = QuantumCircuit(self.num_qubits)
        for param, qubit in zip(encoding, range(self.num_qubits)):
            circuit.rx(param, qubit)

        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]
        return circuit, list(encoding), list(weights), observables

    def get_circuit(self) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
        """Return the full circuit and its metadata for evaluation."""
        return self.circuit, self.encoding_params, self.weight_params, self.observables
