"""Quantum circuit factory with richer ansatz and observables."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuantumClassifierModel:
    """
    Quantum circuit that mirrors the classical interface.

    Returns:
        circuit (QuantumCircuit): Parameterised ansatz.
        encoding (Iterable[ParameterVector]): Feature parameters.
        weights (Iterable[ParameterVector]): Variational parameters.
        observables (List[SparsePauliOp]): Measurement operators.
    """

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int,
                                 entangle: bool = True) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)

        # Feature encoding: RX gates
        for qubit, param in enumerate(encoding):
            circuit.rx(param, qubit)

        # Variational layers
        idx = 0
        for layer in range(depth):
            for qubit in range(num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            if entangle:
                for qubit in range(num_qubits - 1):
                    circuit.cz(qubit, qubit + 1)
                if num_qubits > 2:
                    circuit.cz(num_qubits - 1, 0)

        # Observables: Z on each qubit
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                       for i in range(num_qubits)]

        return circuit, list(encoding), list(weights), observables


__all__ = ["QuantumClassifierModel"]
