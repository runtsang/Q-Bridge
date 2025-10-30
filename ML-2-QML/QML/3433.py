"""Hybrid quantum classifier with data‑uploading encoder and variational layers."""

from __future__ import annotations

from typing import Iterable, Tuple
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class HybridQuantumClassifier:
    """Quantum circuit that mirrors the classical architecture using data‑uploading and a variational ansatz."""
    def __init__(self, num_qubits: int, depth: int) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding_params, self.weight_params, self.observables = self.build_classifier_circuit(num_qubits, depth)

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
        """
        Construct a layered ansatz with explicit data‑uploading encoder.
        Encoding: RX for each qubit per data feature.
        Variational: alternating single‑qubit RY and CZ entanglement.
        Observables: Z on each qubit.
        """
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)
        circuit = QuantumCircuit(num_qubits)

        # Data‑uploading: RX for each qubit with feature parameter
        for idx, qubit in enumerate(range(num_qubits)):
            circuit.rx(encoding[idx], qubit)

        # Variational layers
        w_idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                circuit.ry(weights[w_idx], qubit)
                w_idx += 1
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
        return circuit, list(encoding), list(weights), observables
