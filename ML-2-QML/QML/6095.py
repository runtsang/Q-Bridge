"""Variational quantum classifier with entangling blocks and optional parameterized encoding.

The `QuantumClassifierModel` class provides a Qiskit circuit factory that matches the
interface of the classical helper while adding richer quantum functionality.  It
creates a parameter‑vector encoding, a depth‑controlled ansatz, and a set of
observables corresponding to the Pauli‑Z measurement on each qubit.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuantumClassifierModel:
    """Variational quantum circuit for binary classification."""

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int,
        entangle: bool = True,
    ) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
        """
        Construct a variational ansatz.

        Parameters
        ----------
        num_qubits : int
            Number of qubits / features.
        depth : int
            Number of variational layers.
        entangle : bool, optional
            Whether to insert a CNOT ladder between layers.

        Returns
        -------
        circuit : QuantumCircuit
            The constructed circuit.
        encoding : Iterable[ParameterVector]
            Parameter vectors for feature encoding.
        weights : Iterable[ParameterVector]
            Parameter vectors for variational parameters.
        observables : List[SparsePauliOp]
            Pauli‑Z observables on each qubit.
        """
        # Feature encoding: RX rotation per qubit
        encoding = ParameterVector("x", num_qubits)

        # Variational parameters: one Ry per qubit per layer
        weight_vectors = [ParameterVector(f"theta_{i}", num_qubits) for i in range(depth)]

        circuit = QuantumCircuit(num_qubits)

        # Encode features
        for qubit, param in enumerate(encoding):
            circuit.rx(param, qubit)

        # Variational layers
        for layer, weights in enumerate(weight_vectors):
            for qubit, param in enumerate(weights):
                circuit.ry(param, qubit)
            if entangle:
                # CNOT ladder
                for qb in range(num_qubits - 1):
                    circuit.cx(qb, qb + 1)

        # Observables: Pauli‑Z on each qubit
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

        return circuit, [encoding], weight_vectors, observables


__all__ = ["QuantumClassifierModel"]
