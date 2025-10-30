"""Quantum classifier factory with a data‑re‑uploading ansatz."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class ClassifierFactory:
    """Factory for building a variational quantum classifier.

    The returned tuple mirrors the classical interface:
    (circuit, encoding, weight_params, observables).  The observables
    are Pauli‑Z on each qubit, suitable for a binary classification
    task.
    """

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """Construct a data‑re‑uploading variational circuit.

        Args:
            num_qubits: Number of qubits / feature dimension.
            depth: Number of variational layers.

        Returns:
            circuit: QuantumCircuit instance ready for simulation.
            encoding: List of ParameterVector objects for data encoding.
            weight_params: List of ParameterVector objects for variational parameters.
            observables: List of SparsePauliOp objects measuring Z on each qubit.
        """
        # Data encoding: RX rotations for each feature
        encoding = ParameterVector("x", num_qubits)
        # Variational parameters: one RY per qubit per layer
        weight_params = ParameterVector("theta", num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)

        # First data‑encoding layer
        for qubit in range(num_qubits):
            circuit.rx(encoding[qubit], qubit)

        # Variational layers with entanglement
        for layer in range(depth):
            # Parameterized RY gates
            for qubit in range(num_qubits):
                circuit.ry(weight_params[layer * num_qubits + qubit], qubit)
            # Full‑chain CZ entanglement
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)
            # Optional wrap‑around entanglement
            circuit.cz(num_qubits - 1, 0)

        # Observables: Z on each qubit
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

        return circuit, list(encoding), list(weight_params), observables

__all__ = ["ClassifierFactory"]
