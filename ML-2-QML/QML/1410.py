"""Advanced variational circuit with data re‑uploading and entanglement layers."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuantumClassifierModel:
    """
    Variational circuit for binary classification.
    Implements a data‑re‑uploading ansatz with alternating RX/RZ rotations
    and multi‑qubit entangling layers (CNOT chains). Mirrors the classical
    interface so the two implementations can be swapped.
    """

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable[ParameterVector], List[int], List[SparsePauliOp]]:
        """
        Construct a circuit with `depth` layers of data re‑uploading.
        Each layer consists of:
            - RX(x_i) for each qubit (data encoding)
            - RZ(theta_i) for each qubit (variational parameters)
            - CNOT chain (entanglement)
        The circuit is followed by a measurement observable Z on each qubit.

        Returns:
            circuit: QuantumCircuit instance
            encoding: list of ParameterVector objects for data
            weight_sizes: list containing the number of variational parameters per layer
            observables: list of SparsePauliOp objects measuring Z on each qubit
        """
        # Data encoding parameters
        encoding = ParameterVector("x", num_qubits)
        # Variational parameters per layer
        weight_vectors: List[ParameterVector] = [ParameterVector(f"theta_{l}", num_qubits) for l in range(depth)]

        circuit = QuantumCircuit(num_qubits)

        # Initial encoding
        for qubit, param in zip(range(num_qubits), encoding):
            circuit.rx(param, qubit)

        # Entangling layers with re‑uploading
        for layer_idx, theta_vec in enumerate(weight_vectors):
            # Variational rotations
            for qubit, theta in zip(range(num_qubits), theta_vec):
                circuit.rz(theta, qubit)

            # Entanglement (CNOT chain)
            for qubit in range(num_qubits - 1):
                circuit.cx(qubit, qubit + 1)

            # Re‑upload data
            for qubit, param in zip(range(num_qubits), encoding):
                circuit.rx(param, qubit)

        # Observables: Z on each qubit
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

        weight_sizes = [num_qubits] * depth  # one theta per qubit per layer

        return circuit, list(encoding), weight_sizes, observables


__all__ = ["QuantumClassifierModel"]
