"""Quantum circuit factory with data re‑uploading and entanglement."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuantumClassifierModel:
    """Variational circuit that encodes data, applies entangling layers and re‑uploads
    the data in each block. The circuit returns measurement operators compatible
    with the classical metadata.
    """

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int,
        entangle: bool = True,
    ) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """
        Construct a data‑re‑uploading ansatz with optional entanglement.

        Parameters
        ----------
        num_qubits : int
            Number of qubits / input features.
        depth : int
            Number of data‑re‑uploading layers.
        entangle : bool, optional
            If True, add a CNOT chain after each rotation layer.

        Returns
        -------
        circuit : QuantumCircuit
            The variational circuit.
        encoding : List[ParameterVector]
            Parameters for data encoding (one per qubit).
        weights : List[ParameterVector]
            Variational parameters for each layer.
        observables : List[SparsePauliOp]
            Pauli-Z observables on each qubit.
        """
        encoding = ParameterVector("x", length=num_qubits)
        weights = ParameterVector("theta", length=num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)

        # Initial data encoding
        for q, param in enumerate(encoding):
            circuit.rx(param, q)

        weight_idx = 0
        for _ in range(depth):
            # Re‑upload data
            for q, param in enumerate(encoding):
                circuit.ry(param, q)

            # Variational rotations
            for q in range(num_qubits):
                circuit.rz(weights[weight_idx], q)
                weight_idx += 1

            if entangle:
                # Entangling layer – CNOT chain
                for q in range(num_qubits - 1):
                    circuit.cx(q, q + 1)

        # Observables: measure Z on each qubit
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

        return circuit, [encoding], [weights], observables

    def __init__(self, num_qubits: int, depth: int, **kwargs):
        self.circuit, _, _, _ = self.build_classifier_circuit(num_qubits, depth, **kwargs)


__all__ = ["QuantumClassifierModel"]
