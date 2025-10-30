"""Quantum circuit builder for a data‑uploading classifier.

The module mirrors the API of the classical helper: ``build_classifier_circuit``.
It returns a Qiskit circuit, parameter vectors for data encoding and variational
weights, and a list of Z‑basis observables.

The circuit uses RX rotations for data encoding, followed by a depth‑controlled
variational block of Ry rotations and CZ entanglement.  The observables are
chosen to match the binary classification output of the classical model.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a parameter‑rich data‑uploading circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features).
    depth : int
        Number of variational layers.

    Returns
    -------
    circuit : QuantumCircuit
        The full circuit.
    encoding : Iterable[ParameterVector]
        Parameter vector for the RX data‑encoding gates.
    variational : Iterable[ParameterVector]
        Parameter vector for the Ry variational gates.
    observables : List[SparsePauliOp]
        Z‑basis measurements on each qubit.
    """
    encoding = ParameterVector("x", num_qubits)
    variational = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data‑encoding layer
    for i in range(num_qubits):
        circuit.rx(encoding[i], i)

    # Variational ansatz
    for layer in range(depth):
        for q in range(num_qubits):
            circuit.ry(variational[layer * num_qubits + q], q)
        # Entangle adjacent qubits
        for q in range(num_qubits - 1):
            circuit.cz(q, q + 1)

    # Observables: Z measurement on each qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, encoding, variational, observables

__all__ = ["build_classifier_circuit"]
