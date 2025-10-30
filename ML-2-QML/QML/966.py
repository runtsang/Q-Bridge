"""Quantum classifier factory with tunable entanglement and Pauli‑Z observables."""

from __future__ import annotations

from typing import Iterable, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """
    Construct a layered variational circuit with data‑encoding and configurable entanglement.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features) to encode.
    depth : int
        Number of variational layers.

    Returns
    -------
    circuit : QuantumCircuit
        Parameterized circuit ready for simulation or execution.
    encoding : Iterable
        ParameterVector for data‑encoding rotations.
    weights : Iterable
        ParameterVector for variational parameters.
    observables : list[SparsePauliOp]
        Pauli‑Z observables on each qubit for measurement.
    """
    # Data‑encoding: RX rotations
    encoding = ParameterVector("x", num_qubits)

    # Variational parameters: one Ry per qubit per layer
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Apply encoding
    for q, param in enumerate(encoding):
        circuit.rx(param, q)

    # Variational layers with alternating Ry and CZ entanglement
    idx = 0
    for _ in range(depth):
        # Rotation layer
        for q in range(num_qubits):
            circuit.ry(weights[idx], q)
            idx += 1
        # Entanglement pattern: CZ between nearest neighbours
        for q in range(num_qubits - 1):
            circuit.cz(q, q + 1)
        # Optional long‑range entanglement (next‑nearest neighbours)
        for q in range(num_qubits - 2):
            circuit.cz(q, q + 2)

    # Measurement observables: Pauli‑Z on each qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
