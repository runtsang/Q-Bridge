"""Quantum circuit factory for an incremental data‑uploading classifier.

Supports configurable entanglement patterns and a parameter‑shift compatible ansatz.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    entanglement: str = "full",
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a variational ansatz with data encoding, adjustable entanglement,
    and a parameter‑shift compatible training loop.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features).
    depth : int
        Number of variational layers.
    entanglement : {"full", "nearest"}, optional
        Entanglement pattern between qubits. ``"full"`` connects every pair
        within a layer, ``"nearest"`` connects only adjacent qubits.

    Returns
    -------
    circuit : QuantumCircuit
        The variational circuit.
    encoding : Iterable
        ParameterVector for data encoding (one per qubit).
    weights : Iterable
        ParameterVector for variational parameters.
    observables : list[SparsePauliOp]
        Pauli‑Z measurements on each qubit.
    """
    # Data encoding – RX rotations
    encoding = ParameterVector("x", num_qubits)

    # Variational parameters – one per qubit per layer
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Encode data
    for qubit, param in zip(range(num_qubits), encoding):
        circuit.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        # Rotations
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1

        # Entanglement
        if entanglement == "full":
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    circuit.cz(i, j)
        else:  # nearest‑neighbour
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

    # Measurement observables
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


__all__ = ["build_classifier_circuit"]
