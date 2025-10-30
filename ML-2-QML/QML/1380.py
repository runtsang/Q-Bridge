"""Quantum classifier circuit factory with configurable entanglement and measurement."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def _entangle(circuit: QuantumCircuit, qubits: List[int], style: str) -> None:
    """Apply entangling gates according to style."""
    if style == "linear":
        for q in range(len(qubits) - 1):
            circuit.cz(qubits[q], qubits[q + 1])
    elif style == "full":
        for i in range(len(qubits)):
            for j in range(i + 1, len(qubits)):
                circuit.cz(qubits[i], qubits[j])
    else:
        raise ValueError(f"Unsupported entanglement style: {style}")


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    entanglement: str = "linear",
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a variational classifier circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features).
    depth : int
        Number of variational layers.
    entanglement : str, default "linear"
        Entanglement pattern: "linear" or "full".

    Returns
    -------
    circuit : QuantumCircuit
        The constructed circuit.
    encoding : Iterable
        ParameterVector for feature encoding.
    weights : Iterable
        ParameterVector for variational parameters.
    observables : List[SparsePauliOp]
        Pauli‑Z operators on each qubit for expectation measurements.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Feature encoding (RX)
    for qubit, param in zip(range(num_qubits), encoding):
        circuit.rx(param, qubit)

    # Variational layers
    weight_idx = 0
    for _ in range(depth):
        # Rotation layer
        for qubit in range(num_qubits):
            circuit.ry(weights[weight_idx], qubit)
            weight_idx += 1
        # Entanglement
        _entangle(circuit, list(range(num_qubits)), entanglement)

    # Measurement observables
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
