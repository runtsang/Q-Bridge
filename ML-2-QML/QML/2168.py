"""Enhanced quantum classifier circuit with entangling readouts and configurable depth.

The API matches the original quantum helper, but now includes:
* Feature‑map with Rx encoding.
* Two entanglement patterns: nearest‑neighbour and all‑to‑all.
* Variational layers with Ry rotations.
* Readouts: single‑qubit Z and two‑qubit ZZ operators for richer measurement statistics.
"""
from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class QuantumClassifierModel:
    """Factory for a variational quantum classifier.

    The static method ``build_classifier_circuit`` returns the same tuple
    that the classical counterpart expects:  ``(circuit, encoding, weights, observables)``.
    The implementation now supports configurable entanglement and
    richer measurement observables.
    """
    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int,
        entanglement: str = "nearest",
        enable_zz: bool = True,
    ) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """
        Construct a layered ansatz with explicit encoding and variational parameters.

        Parameters
        ----------
        num_qubits : int
            Number of qubits used for the classifier.
        depth : int
            Number of variational layers.
        entanglement : {"nearest", "full"}, optional
            Entanglement pattern between qubits.
        enable_zz : bool, optional
            Whether to add ZZ measurement operators.

        Returns
        -------
        Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]
            * ``circuit`` – the variational quantum circuit.
            * ``encoding`` – list of ParameterVector objects for feature encoding.
            * ``weights`` – list of ParameterVector objects for variational parameters.
            * ``observables`` – list of SparsePauliOp measurement operators.
        """
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)

        # Feature encoding
        for qubit, param in enumerate(encoding):
            circuit.rx(param, qubit)

        # Variational layers
        weight_idx = 0
        for _ in range(depth):
            # local rotations
            for qubit in range(num_qubits):
                circuit.ry(weights[weight_idx], qubit)
                weight_idx += 1
            # entanglement
            if entanglement == "nearest":
                for qubit in range(num_qubits - 1):
                    circuit.cz(qubit, qubit + 1)
            elif entanglement == "full":
                for qubit in range(num_qubits):
                    for other in range(qubit + 1, num_qubits):
                        circuit.cz(qubit, other)
            else:
                raise ValueError("Unsupported entanglement pattern")

        # Observables
        observables: List[SparsePauliOp] = []
        # single‑qubit Zs
        for i in range(num_qubits):
            observables.append(SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)))
        # optional ZZ terms for richer readout
        if enable_zz:
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    pauli = "I" * i + "Z" + "I" * (j - i - 1) + "Z" + "I" * (num_qubits - j - 1)
                    observables.append(SparsePauliOp(pauli))

        return circuit, list(encoding), list(weights), observables

__all__ = ["QuantumClassifierModel"]
