"""Advanced quantum classifier circuit with controlled‑phase encoding and ring CZ entanglement."""

from __future__ import annotations

from typing import Iterable, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class QuantumClassifierModel:
    """
    Quantum circuit builder that mirrors the classical interface.
    The circuit contains a controlled‑phase encoding followed by
    depth‑dependent variational rotations and ring‑shaped CZ gates.
    """

    def __init__(self, num_qubits: int, depth: int):
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding, self.weights, self.observables = self.build_classifier_circuit()

    def build_classifier_circuit(
        self,
    ) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
        """
        Construct the variational circuit.

        Returns:
            circuit: QuantumCircuit instance.
            encoding: list of ParameterVector for data encoding.
            weights: list of ParameterVector for variational parameters.
            observables: list of PauliZ observables on each qubit.
        """
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        circuit = QuantumCircuit(self.num_qubits)

        # Controlled‑phase encoding: RX followed by RZ on each qubit
        for qubit in range(self.num_qubits):
            circuit.rx(encoding[qubit], qubit)
            circuit.rz(encoding[qubit], qubit)

        for layer in range(self.depth):
            # Variational rotations
            for qubit in range(self.num_qubits):
                circuit.ry(weights[layer * self.num_qubits + qubit], qubit)
            # Ring CZ entanglement
            for qubit in range(self.num_qubits):
                circuit.cz(qubit, (qubit + 1) % self.num_qubits)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]

        return circuit, list(encoding), list(weights), observables

__all__ = ["QuantumClassifierModel"]
