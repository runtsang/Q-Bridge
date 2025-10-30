"""Hybrid classifier module for the quantum side.

Provides a variational circuit that matches the classical interface.
"""

from __future__ import annotations

from typing import Iterable, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class HybridClassifier:
    @staticmethod
    def build_classifier_circuit(
        num_qubits: int,
        depth: int,
    ) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], list[SparsePauliOp]]:
        """Construct a layered variational classifier."""
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)

        # Data‑encoding layer
        for i, param in enumerate(encoding):
            circuit.rx(param, i)

        # Variational layers
        weight_idx = 0
        for _ in range(depth):
            for q in range(num_qubits):
                circuit.ry(weights[weight_idx], q)
                weight_idx += 1
            # Entanglement
            for q in range(num_qubits - 1):
                circuit.cz(q, q + 1)
            if num_qubits > 2:
                circuit.cz(num_qubits - 1, 0)

        # Observables
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

        return circuit, list(encoding), list(weights), observables


__all__ = ["HybridClassifier"]
