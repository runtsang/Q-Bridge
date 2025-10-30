"""Quantum classifier with shared parameters and ring entanglement.

The function `build_classifier_circuit` mirrors the classical counterpart
but builds a parametric circuit.  It now uses a shared ParameterVector
for the variational rotations and a ring of CZ gates for entanglement.
The returned metadata (`encoding`, `weights`, `observables`) keeps the
original API shape.
"""

from __future__ import annotations

from typing import Iterable, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """Construct a hybrid variational circuit.

    Returns:
        circuit     – QuantumCircuit object.
        encoding    – list of input encoding parameters.
        weights     – list of shared variational parameters.
        observables – list of SparsePauliOp measurement operators.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits)  # shared across layers

    circuit = QuantumCircuit(num_qubits)

    # Feature encoding
    for idx, qubit in enumerate(range(num_qubits)):
        circuit.rx(encoding[idx], qubit)

    # Variational layers with shared parameters and ring entanglement
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[qubit], qubit)
        for qubit in range(num_qubits):
            circuit.cz(qubit, (qubit + 1) % num_qubits)

    # Observables: Z on each qubit
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

    return circuit, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
