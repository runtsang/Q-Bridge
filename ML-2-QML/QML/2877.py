"""
Quantum counterpart of the HybridClassifier.
Implements a QCNN‑style ansatz with a data‑encoding layer
and a set of single‑qubit Z observables, matching the classical
output neurons.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import numpy as np

__all__ = ["HybridClassifier"]


class HybridClassifier:
    """Quantum hybrid classifier with QCNN‑style ansatz."""

    def __init__(self, num_qubits: int, depth: int = 3) -> None:
        """
        Parameters
        ----------
        num_qubits: int
            Number of qubits (must match the classical input dimension).
        depth: int
            Number of convolutional layers in the ansatz.
        """
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

    def _build_circuit(self):
        """
        Builds a QCNN‑style circuit:
        - Rx rotations for data encoding.
        - Ry rotations and CZ entanglement per convolutional layer.
        - Single‑qubit Z observables per qubit.
        """
        # Data‑encoding layer
        encoding = ParameterVector("x", self.num_qubits)

        # Variational parameters
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        circ = QuantumCircuit(self.num_qubits)

        # Encode data
        for qubit, param in zip(range(self.num_qubits), encoding):
            circ.rx(param, qubit)

        # Convolution layers with CZ entanglement
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                circ.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                circ.cz(qubit, qubit + 1)

        # Observables: single‑qubit Z on each qubit
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]

        return circ, list(encoding), list(weights), observables

    def get_circuit(self) -> QuantumCircuit:
        """Return the full QCNN‑style circuit."""
        return self.circuit

    def get_parameters(self) -> list[ParameterVector]:
        """Return the encoding and variational parameters."""
        return self.encoding + self.weights

    def get_observables(self) -> list[SparsePauliOp]:
        """Return the measurement observables."""
        return self.observables
