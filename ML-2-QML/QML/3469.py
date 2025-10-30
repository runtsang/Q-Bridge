"""Hybrid quantum classifier factory using Qiskit.

This module provides a circuit that mirrors the classical feed‑forward
network and an estimator capable of evaluating expectation values.
The API is intentionally compatible with the classical side so that
experiments can switch between the two back‑ends seamlessly.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

from.FastBaseEstimator import FastBaseEstimator

# ----------------------------------------------------------------------
# Quantum circuit construction
# ----------------------------------------------------------------------
def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a layered ansatz that matches the classical network
    signature.  The function returns (circuit, encoding, weights,
    observables) where ``encoding`` and ``weights`` are ParameterVectors
    that can be bound to the circuit.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables – Z on each qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, encoding, weights, observables


# ----------------------------------------------------------------------
# Hybrid model definition
# ----------------------------------------------------------------------
class HybridClassifier:
    """
    Quantum circuit wrapper that provides the same API surface as the
    classical HybridClassifier.  ``evaluate`` runs the circuit for a
    batch of parameter sets and returns expectation values for the
    predefined observables.
    """

    def __init__(self, num_qubits: int, depth: int) -> None:
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)

    def evaluate(self, parameter_sets: List[List[float]]) -> List[List[complex]]:
        estimator = FastBaseEstimator(self.circuit)
        return estimator.evaluate(self.observables, parameter_sets)


# ----------------------------------------------------------------------
# Estimator utilities
# ----------------------------------------------------------------------
class HybridEstimator(FastBaseEstimator):
    """
    Thin wrapper around FastBaseEstimator that matches the naming
    scheme of the classical side.  It accepts a QuantumCircuit and
    returns expectation values for a batch of parameters.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        super().__init__(circuit)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: List[List[float]],
    ) -> List[List[complex]]:
        return super().evaluate(observables, parameter_sets)


__all__ = ["build_classifier_circuit", "HybridClassifier", "HybridEstimator"]
