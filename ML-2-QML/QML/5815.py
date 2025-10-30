"""Quantum classifier factory with a hardware‑efficient ansatz.

The class offers a variational circuit builder that includes optional
wrap‑around entanglement and a richer set of measurement operators.
This mirrors the classical API while providing a meaningful quantum
contribution for hybrid workflows.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuantumClassifierModel:
    """
    Factory for quantum classification circuits.
    """

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int,
        *,
        entangle: bool = True,
    ) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
        """
        Construct a hardware‑efficient variational circuit with optional
        wrap‑around entanglement.

        Parameters
        ----------
        num_qubits : int
            Number of qubits (features) in the circuit.
        depth : int
            Number of variational layers.
        entangle : bool, optional
            If ``True`` a CZ entanglement pattern that wraps around the
            qubit chain is applied after each rotation layer.  Defaults to
            ``True``.

        Returns
        -------
        circuit : qiskit.QuantumCircuit
            The assembled variational circuit.
        encoding : Iterable[ParameterVector]
            Parameter vectors for data encoding.
        weights : Iterable[ParameterVector]
            Parameter vectors for the variational layers.
        observables : List[qiskit.quantum_info.SparsePauliOp]
            Pauli‑Z observables on each qubit, suitable for expectation‑value
            evaluation.
        """
        # Data encoding: RX rotations
        encoding = ParameterVector("x", num_qubits)
        # Variational parameters: one θ per qubit per layer
        weights = ParameterVector("theta", num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)

        # Encoding layer
        for qubit in range(num_qubits):
            circuit.rx(encoding[qubit], qubit)

        # Variational layers
        weight_index = 0
        for _ in range(depth):
            # Single‑qubit rotations
            for qubit in range(num_qubits):
                circuit.ry(weights[weight_index], qubit)
                weight_index += 1

            # Entangling layer
            if entangle:
                for qubit in range(num_qubits - 1):
                    circuit.cz(qubit, qubit + 1)
                # Wrap‑around entanglement to make the graph fully connected
                circuit.cz(num_qubits - 1, 0)

        # Measurement observables: Z on each qubit
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

        return circuit, [encoding], [weights], observables


__all__ = ["QuantumClassifierModel"]
